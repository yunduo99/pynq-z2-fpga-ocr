# -*- coding: utf-8 -*-
"""
识别终端 v2 —— 支持「识别后端」切换的独立终端
============================================================
与原终端(pynq_ocr.py / 识别终端.ipynb)完全独立, 原终端不受影响。

特点:
  * 复用原终端已验证的硬件流水线(PL 二值化+形态学、条码、ROI)
  * **可切换识别后端**:
      backend='local'  → 板卡 PS 端 tesseract 识别(默认, 符合项目规格)
      backend='remote' → 把文字块发到 PC, 用 PaddleOCR 等更强引擎识别(准确率更高)
  * 一键接口: recognize() 自动判断图片/视频
  * 结果可导出 JSON / CSV

用法:
    import pynq_ocr2 as po2
    po2.recognize()                       # 自动处理最新素材(本地识别)
    po2.set_backend('remote', '192.168.2.1')   # 切到 PC 识别(需在 PC 跑 pc_ocr_server.py)
    po2.recognize('视频.mp4', per_sec=1.0)     # 每秒识别一次
"""
import os, sys, time, json, glob, socket, struct
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pynq_ocr as po                     # 复用经过验证的流水线

OUTDIR = po.OUTDIR
VERSION = 'v2-2026.09.29'

_state = {'backend': 'local', 'host': '192.168.2.1', 'port': 9988, 'last': []}


# ============================================================
#  识别后端
# ============================================================
def _local_ocr(img):
    """板卡本地识别(tesseract)"""
    lines, ms = po._T.ocr_img(img)
    return lines, ms


def _remote_ocr(img, timeout=180):
    """把文字块发到 PC 识别(PC 端需运行 pc_ocr_server.py)"""
    t0 = time.perf_counter()
    h, w = img.shape
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        return [], 0.0
    data = buf.tobytes()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((_state['host'], _state['port']))
        s.sendall(struct.pack('!II', w, h) + struct.pack('!I', len(data)) + data)
        # 收结果: 4 字节长度 + JSON
        hdr = _recv_exact(s, 4)
        n = struct.unpack('!I', hdr)[0]
        payload = _recv_exact(s, n)
        res = json.loads(payload.decode('utf-8'))
        lines = [l for l in (res.get('texts') or []) if l and l.strip()]
        return lines, (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        po._log(f"  ⚠️ 远程识别失败({type(e).__name__}: {e}), 回退本地识别")
        return _local_ocr(img)
    finally:
        try:
            s.close()
        except Exception:
            pass


def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError('连接中断')
        buf += chunk
    return buf


def set_backend(name='local', host=None, port=None, quiet=False):
    """切换识别后端: 'local'(板卡) / 'remote'(PC)
       remote 模式需在 PC 上运行 pc_ocr_server.py"""
    name = (name or 'local').lower()
    if name not in ('local', 'remote'):
        po._log(f"⚠️ 未知后端 {name!r}, 可选 'local' / 'remote'")
        return
    _state['backend'] = name
    if host:
        _state['host'] = host
    if port:
        _state['port'] = int(port)
    if name == 'remote':
        po._T.ocr_hook = _remote_ocr
        if not quiet:
            po._log(f"识别后端 → remote (PC {_state['host']}:{_state['port']})")
    else:
        po._T.ocr_hook = None
        if not quiet:
            po._log("识别后端 → local (板卡 PS 端 tesseract)")


def test_remote(host=None, port=None):
    """测试与 PC 服务的连通性"""
    if host:
        _state['host'] = host
    if port:
        _state['port'] = int(port)
    po._log(f"测试连接 {_state['host']}:{_state['port']} ...")
    try:
        s = socket.socket()
        s.settimeout(8)
        s.connect((_state['host'], _state['port']))
        s.sendall(struct.pack('!III', 0, 0, 0))     # 空帧 = 心跳
        hdr = _recv_exact(s, 4)
        n = struct.unpack('!I', hdr)[0]
        res = json.loads(_recv_exact(s, n).decode('utf-8'))
        s.close()
        po._log(f"  ✅ 连接成功, PC 端引擎: {res.get('engine', '?')}")
        return True
    except Exception as e:
        po._log(f"  ❌ 连接失败: {type(e).__name__}: {e}")
        po._log(f"     请确认 PC 上已运行: python pc_ocr_server.py")
        return False


# ============================================================
#  一键识别
# ============================================================
def _newest(exts):
    cands = []
    for d in (os.path.join(OUTDIR, 'docs'), OUTDIR):
        for e in exts:
            cands += glob.glob(os.path.join(d, '*.' + e))
    cands = [c for c in cands if '_crop' not in os.path.basename(c)]
    if not cands:
        return None
    cands.sort(key=os.path.getmtime, reverse=True)
    return cands[0]


def recognize(path='', per_sec=1.0, max_seconds=180, crop=None, accurate=True,
              drop_latin=True, tile=None, do_ocr=True, max_roi=3, rotate='auto',
              save=True):
    """一键识别: 自动判断是图片还是视频
         path 留空 -> 自动选择最新素材
         save=True -> 结果保存到 outcomes/ 目录(JSON)"""
    if not path:
        p = _newest(['jpg', 'jpeg', 'png', 'bmp', 'webp'])
        v = _newest(['mp4', 'avi', 'mov', 'mkv'])
        # 选更新的那个
        if p and v:
            path = p if os.path.getmtime(p) > os.path.getmtime(v) else v
        else:
            path = p or v
        if not path:
            po._log("❌ 目录里没有图片或视频"); return None
        po._log(f"  (自动选择最新素材: {os.path.basename(path)})")

    is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    po._log(f"[{VERSION}] 后端={_state['backend']}  "
            f"{'视频(每秒%.2g次)' % per_sec if is_video else '图片'}")
    if is_video:
        res = po._T.video(path, per_sec=per_sec, max_seconds=max_seconds,
                          do_ocr=do_ocr, accurate=accurate, max_roi=max_roi,
                          rotate=rotate, drop_latin=drop_latin, tile=tile)
        out = _summarize_video(res, path)
    else:
        res = po._T.image(path, crop=crop, accurate=accurate, do_ocr=do_ocr,
                          max_roi=max_roi, rotate=rotate, drop_latin=drop_latin, tile=tile)
        out = _summarize_image(res, path)
    _state['last'] = out
    if save and out:
        save_result(out)
    return out


def _summarize_image(R, path):
    if not R:
        return None
    return {'file': os.path.basename(path), 'type': 'image',
            'backend': _state['backend'],
            'bars': [{'type': t, 'data': d, 'rect': list(r)} for t, d, r in R.get('bars', [])],
            'texts': R.get('texts', []),
            'isbn': _isbn_of(R.get('bars', [])),
            'timing_ms': {'PL': round(R.get('pl', 0), 2),
                          'barcode': round(R.get('t_bar', 0), 1),
                          'ocr': round(R.get('t_ocr', 0), 1),
                          'total': round(R.get('total', 0), 1)}}


def _summarize_video(allR, path):
    if not allR:
        return None
    bc, tc = {}, {}
    bars = []
    for R in allR:
        for b in R.get('bars', []):
            key = (b[0], b[1])
            bc[key] = bc.get(key, 0) + 1
            if key not in [(x[0], x[1]) for x in bars]:
                bars.append(b)
        for t in R.get('texts', []):
            tc[t] = tc.get(t, 0) + 1
    n = len(allR)
    return {'file': os.path.basename(path), 'type': 'video', 'backend': _state['backend'],
            'samples': n,
            'bars': [{'type': t, 'data': d, 'count': bc.get((t, d), 0)} for t, d, _r in bars],
            'texts': [{'text': t, 'count': c} for t, c in sorted(tc.items(), key=lambda kv: -kv[1])],
            'isbn': _isbn_of(bars),
            'per_sample_ms': round(sum(R.get('total', 0) for R in allR) / n, 1)}


def _isbn_of(bars):
    for t, d, *_ in bars:
        if t == 'EAN13' and str(d)[:3] in ('978', '979') and len(str(d)) == 13:
            d = str(d)
            return f"{d[:3]}-{d[3:6]}-{d[6:9]}-{d[9:12]}-{d[12:]}"
    return None


def save_result(out, name=''):
    """结果存成 JSON(默认 outcomes/ 目录)"""
    d = os.path.join(OUTDIR, 'outcomes')
    os.makedirs(d, exist_ok=True)
    if not name:
        base = os.path.splitext(out['file'])[0]
        name = f"{base}_{int(time.time())}.json"
    p = os.path.join(d, name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    po._log(f"  结果已保存: {p}")
    return p


def show_last():
    """打印上次识别结果(结构化)"""
    out = _state.get('last')
    if not out:
        po._log("(还没有识别结果)"); return
    po._log(json.dumps(out, ensure_ascii=False, indent=2))


def status():
    po._log(f"识别终端 {VERSION}")
    po._log(f"  识别后端: {_state['backend']}"
            + (f"  → PC {_state['host']}:{_state['port']}" if _state['backend'] == 'remote' else "  (板卡 PS tesseract)"))
    po._log(f"  流水线: 复用 pynq_ocr {po.VERSION}")
    po._T.status()
