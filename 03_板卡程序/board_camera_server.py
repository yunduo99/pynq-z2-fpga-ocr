# -*- coding: utf-8 -*-
"""
板卡端：实时识别服务
============================================================
配合 PC 端 pc_camera_client.py 使用。

协议(与 pc_ocr_server.py 一致):
  客户端 → 板卡:  [4B w][4B h][4B n] + n 字节数据
      n>0 且 w>0 : 一帧图像(JPEG)
      w==0, n==0 : 心跳
      w==0, n>0  : 控制命令(JSON, 见下)
  板卡 → 客户端:  [4B len] + len 字节 JSON

控制命令(JSON):
  {"mode":"barcode"}           只做条码(最快)
  {"mode":"ocr"}               条码 + 文字(默认, 快速模式)
  {"mode":"ocr_accurate"}      条码 + 文字(高精度, 慢)
  {"backend":"local|remote"}   识别后端切换
  {"gate":true|false}          画面未变化时跳过 OCR
  {"crop":[x0,y0,x1,y1]}       只识别该区域(可选)
  {"reset":true}               重新加载硬件

返回 JSON:
  {"ok":true, "bars":[{"type","data","rect"}], "texts":[...],
   "isbn":"...", "timing":{"pl","t_bar","t_ocr","total"}, "size":[w,h], "gate":bool}

启动:
  python3 board_camera_server.py            # 默认端口 9999
"""
import os, sys, json, time, socket, struct
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PORT = int(os.environ.get('PORT', '9999'))
HOST = os.environ.get('HOST', '0.0.0.0')

# ---- 运行状态 ----
ST = {
    'mode': 'ocr',          # barcode / ocr / ocr_accurate
    'backend': 'local',
    'layout': 'auto',
    'ocr_budget_ms': 0,
    'ocr_workers': 2,
    'target_ms': 1000,
    'pc_host': '192.168.2.1',
    'pc_port': 9989,
    'gate': True,
    'crop': None,
    'drop_latin': False,    # 摄像头场景默认保留英文(不要丢 Hello 这类纯英文)
    'strict': True,         # 严格文字形状校验(滤掉背景暗色物体); 可按 t 键关闭
    'lang': 'chi_sim',      # ⚠️ 必须用纯 chi_sim; 'chi_sim+eng' 会让中文变拉丁乱码
    'prev_thumb': None,
    'n': 0,
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError('连接中断')
        buf += c
    return buf


def send_json(sock, obj):
    p = json.dumps(obj, ensure_ascii=False).encode('utf-8')
    sock.sendall(struct.pack('!I', len(p)) + p)


def handle_command(cmd):
    """处理控制命令"""
    if 'layout' in cmd and cmd['layout'] not in ('auto', 'document', 'scene', 'legacy'):
        raise ValueError('layout 必须是 auto/document/scene/legacy')
    if 'mode' in cmd and cmd['mode'] not in ('barcode', 'ocr', 'ocr_accurate'):
        raise ValueError('未知识别模式')
    if 'backend' in cmd and cmd['backend'] not in ('local', 'pc', 'remote'):
        raise ValueError('backend 必须为 local/pc/remote')
    if 'ocr_workers' in cmd and cmd['ocr_workers'] not in (1, 2):
        raise ValueError('ocr_workers 必须是 1 或 2')
    if 'target_ms' in cmd and not 100 <= int(cmd['target_ms']) <= 180000:
        raise ValueError('target_ms 超出范围')
    if 'ocr_budget_ms' in cmd:
        budget = int(cmd['ocr_budget_ms'])
        if budget != 0 and not 1000 <= budget <= 180000:
            raise ValueError('ocr_budget_ms 必须为 0 或 1000..180000')
    import pynq_ocr as po
    # Every setting change invalidates both text and bounding-box caches.
    ST['prev_thumb'] = None
    po._T._smart_cache = None
    if 'layout' in cmd:
        ST['layout'] = cmd['layout']
    if 'ocr_budget_ms' in cmd:
        ST['ocr_budget_ms'] = int(cmd['ocr_budget_ms'])
    if 'mode' in cmd:
        ST['mode'] = str(cmd['mode'])
    if 'backend' in cmd:
        import pynq_ocr2 as po2
        po2.set_backend('remote' if cmd['backend'] == 'remote' else 'local', quiet=True)
        ST['backend'] = cmd['backend']
    for name in ('ocr_workers', 'target_ms', 'pc_host', 'pc_port'):
        if name in cmd:
            ST[name] = cmd[name]
    po._T.ocr_backend = ST['backend']
    po._T.ocr_workers = ST['ocr_workers']
    if 'ocr_workers' in cmd and getattr(po._T, 'ocr', None):
        from performance_runtime import prepare_runtime
        prepare_runtime(po._T)
    if 'gate' in cmd:
        ST['gate'] = bool(cmd['gate'])
        if not ST['gate']:
            ST['prev_thumb'] = None
    if 'crop' in cmd:
        c = cmd['crop']
        ST['crop'] = tuple(int(v) for v in c) if c else None
    if 'drop_latin' in cmd:
        ST['drop_latin'] = bool(cmd['drop_latin'])
    if 'strict' in cmd:
        ST['strict'] = bool(cmd['strict'])
    if 'lang' in cmd:
        import pynq_ocr as po
        po._T.set_lang(str(cmd['lang']))
        ST['lang'] = str(cmd['lang'])
    if cmd.get('reset'):
        import pynq_ocr as po
        po._T.ready = False
        po._T.init(force=True)
        ST['prev_thumb'] = None
    log(f"命令生效: {cmd} → mode={ST['mode']} backend={ST['backend']} "
        f"gate={ST['gate']} crop={ST['crop']}")


def process_frame(jpg_bytes, w, h):
    """识别一帧, 返回结果 dict"""
    import cv2
    import pynq_ocr as po

    arr = np.frombuffer(jpg_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {'ok': False, 'error': 'JPEG 解码失败'}
    input_height, input_width = frame.shape[:2]
    if ST['crop']:
        x0, y0, x1, y1 = ST['crop']
        H, W = frame.shape[:2]
        x0, x1 = max(0, min(x0, W - 1)), max(1, min(x1, W))
        y0, y1 = max(0, min(y0, H - 1)), max(1, min(y1, H))
        cw, ch = x1 - x0, y1 - y0
        # 防御: 裁剪区域过小(如超出画面被钳成 1 像素) 时忽略裁剪并告警
        if cw < 24 or ch < 12:
            log(f"⚠️ 裁剪区域 {ST['crop']} 在 {W}x{H} 画面上只剩 {cw}x{ch}, 已忽略")
            ST['crop'] = None
        else:
            frame = frame[y0:y1, x0:x1]

    do_ocr = ST['mode'] != 'barcode'
    accurate = ST['mode'] == 'ocr_accurate'
    t0 = time.perf_counter()
    po._T.ocr_layout = ST['layout']
    po._T.ocr_budget_ms = ST['ocr_budget_ms']
    for name in ('ocr_workers', 'target_ms', 'pc_host', 'pc_port'):
        setattr(po._T, name, ST[name])
    po._T.ocr_backend = ST['backend']
    R, thumb = po._T.process(frame, do_ocr=do_ocr,
                             prev_thumb=ST['prev_thumb'] if ST['gate'] else None,
                             gate_th=10.0, max_roi=4, cropmax=450,
                             accurate=accurate, drop_latin=ST['drop_latin'],
                             strict_text=ST['strict'],
                             per_roi=(ST['mode'] != 'barcode'))   # 逐框识别, 结果与框一一对应
    dt = (time.perf_counter() - t0) * 1000
    if ST['gate']:
        ST['prev_thumb'] = thumb
    if R.get('error'):
        return {'ok': False, 'error': R['error'], 'complete': False, 'profile': R.get('profile')}

    bars = [{'type': t, 'data': d, 'rect': [int(v) for v in r]} for t, d, r in R.get('bars', [])]
    rois = [[int(v) for v in r] for r in R.get('rois', [])]
    isbn = None
    for b in bars:
        if b['type'] == 'EAN13' and str(b['data'])[:3] in ('978', '979') and len(str(b['data'])) == 13:
            s = str(b['data'])
            isbn = f"{s[:3]}-{s[3:6]}-{s[6:9]}-{s[9:12]}-{s[12:]}"
            break
    return {
        'ok': True,
        'complete': R.get('complete', True),
        'warnings': R.get('warnings', []),
        'layout': R.get('layout', ST['layout']),
        'ocr_calls': R.get('ocr_calls'),
        'confidence': R.get('conf', -1),
        'profile': R.get('profile'),
        'parallel': R.get('parallel'),
        'pc_timing': R.get('pc_timing'),
        'neural_engine': R.get('neural_engine'),
        'target_ms': R.get('target_ms'),
        'fresh_target_met': R.get('fresh_target_met'),
        'detected_lines': R.get('detected_lines'),
        'barcode_checked': R.get('barcode_checked', True),
        'bars': bars,
        'texts': R.get('texts', []),
        'isbn': isbn,
        'gate_skipped': bool(R.get('cached')),
        'rois': rois,
        'roi_src': R.get('roi_src', ''),
        'roi_texts': R.get('roi_texts') or [],   # 与 rois 一一对应的文字
        'crop': list(ST['crop']) if ST['crop'] else None,
        'size': [int(frame.shape[1]), int(frame.shape[0])],
        # 所有框(条码/文字)都已统一到"收到的这一帧"的坐标, 客户端据此缩放显示
        'proc_size': [int(input_width), int(input_height)],
        'timing': {'pl': round(R.get('pl', 0), 2),
                   't_bar': round(R.get('t_bar', 0), 1),
                   't_roi': round(R.get('t_roi', 0), 1),
                   't_ocr': round(R.get('t_ocr', 0), 1),
                   'total': round(R.get('total', 0), 1),
                   'server': round(dt, 1)},
        'mode': ST['mode'], 'backend': ST['backend'], 'lang': ST['lang'],
    }


def serve():
    import pynq_ocr as po
    log("=" * 62)
    log("  板卡端实时识别服务")
    log("=" * 62)
    log("初始化硬件 ...")
    po._T.morph_mode = 1
    po._T.init()
    log(f"  就绪: {po.VERSION}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(2)
    log(f"  监听 {HOST}:{PORT}   等待 PC 连接 ...")

    while True:
        conn, addr = srv.accept()
        log(f"PC 已连接: {addr[0]}:{addr[1]}")
        # The supplied PC client opens a NEW connection for every frame/command.
        # Preserve settings and the recognized-frame cache across those requests.
        # This is a single-operator service; commands explicitly clear caches.
        try:
            while True:
                hdr = recv_exact(conn, 12)
                w, h, n = struct.unpack('!III', hdr)
                if n == 0:
                    if w == 0:
                        send_json(conn, {'ok': True, 'engine': 'pynq-pl',
                                         'version': po.VERSION, 'mode': ST['mode'],
                                         'backend': ST['backend']})
                        continue
                    data = b''
                else:
                    data = recv_exact(conn, n)
                if w == 0:                       # 控制命令
                    try:
                        cmd = json.loads(data.decode('utf-8'))
                        handle_command(cmd)
                        send_json(conn, {'ok': True, 'mode': ST['mode'],
                                         'backend': ST['backend'], 'gate': ST['gate'],
                                         'crop': list(ST['crop']) if ST['crop'] else None})
                    except Exception as e:
                        send_json(conn, {'ok': False, 'error': f'命令解析失败: {e}'})
                    continue
                # 图像帧
                try:
                    res = process_frame(data, w, h)
                except Exception as e:
                    res = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
                send_json(conn, res)
                ST['n'] += 1
                if ST['n'] % 10 == 0:
                    t = res.get('timing', {})
                    log(f"已处理 {ST['n']} 帧 | 最近: PL {t.get('pl')}ms 条码 {t.get('t_bar')}ms "
                        f"OCR {t.get('t_ocr')}ms 合计 {t.get('total')}ms")
        except (ConnectionError, OSError) as e:
            log(f"PC 断开: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == '__main__':
    serve()
