# -*- coding: utf-8 -*-
"""
PC 端：实时识别客户端（USB 摄像头 → 板卡 → 结果叠加显示）
============================================================
用法:
    python pc_camera_client.py                       # 用默认摄像头(0)
    python pc_camera_client.py --cam 1               # 指定摄像头编号
    python pc_camera_client.py --source 视频.mp4      # 用视频文件模拟摄像头(测试用)
    python pc_camera_client.py --host 192.168.2.99   # 板卡地址(默认)
    python pc_camera_client.py --test 20             # 无界面模式, 只发 20 帧看结果

按键:
    q 退出    b 只做条码    o 条码+文字    a 高精度OCR
    g 开关门控    r 重载硬件    s 保存截图

依赖: pip install opencv-python numpy
"""
import argparse, json, socket, struct, sys, threading, time, os
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')      # 必须在 import cv2 之前
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
import numpy as np
import cv2

# ---- 关闭 OpenCV 的刷屏警告 ----
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# ---- 中文渲染(PIL; cv2.putText 不支持中文) ----
_CN_FONT = None
def _init_font():
    """加载中文字体(微软雅黑), 失败返回 None"""
    global _CN_FONT
    try:
        from PIL import ImageFont
        for fp in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
                   r'C:\Windows\Fonts\simsun.ttc'):
            if os.path.isfile(fp):
                _CN_FONT = ImageFont.truetype(fp, 22)
                return _CN_FONT
    except Exception:
        pass
    return None


def put_cn(img, text, pos, color=(0, 255, 255), size=22):
    """在图上画中文(cv2.putText 画不了中文, 用 PIL)"""
    global _CN_FONT
    if _CN_FONT is None:
        _init_font()
    if _CN_FONT is None:                      # 没字体就退回英文渲染
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return img
    try:
        from PIL import Image, ImageDraw, ImageFont
        f = _CN_FONT if size == 22 else ImageFont.truetype(_CN_FONT.path, size)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        d.text(pos, text, font=f, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return img


# ---- Windows 控制台中文 ----
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

HOST, PORT = '192.168.2.99', 9999
SEND_W = 1280          # 发送给板卡的处理宽度(≤1920 ✓, 保证坐标对齐 ✓)


# ============================================================
#  与板卡的通信线程(请求-响应, 自动节流)
# ============================================================
class BoardLink(threading.Thread):
    def __init__(self, host, port, on_result=None):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.on_result = on_result
        self.lock = threading.Lock()
        self.pending = None          # (seq, jpg_bytes)
        self.result = None           # 最近一次识别结果
        self.seq = 0
        self.done_seq = -1
        self.connected = False
        self.stop_flag = False
        self.last_error = ''
        self.rtt_ms = 0.0
        self.frames = 0

    def submit(self, jpg):
        with self.lock:
            self.seq += 1
            self.pending = (self.seq, jpg)

    def snapshot(self):
        with self.lock:
            return self.result, self.connected, self.rtt_ms, self.frames, self.last_error

    def send_cmd(self, cmd):
        """发送控制命令"""
        try:
            s = socket.create_connection((self.host, self.port), timeout=10)
            p = json.dumps(cmd).encode()
            s.sendall(struct.pack('!III', 0, 0, len(p)) + p)
            n = struct.unpack('!I', _recv_exact(s, 4))[0]
            res = json.loads(_recv_exact(s, n).decode('utf-8'))
            s.close()
            return res
        except Exception as e:
            return {'ok': False, 'error': f'{type(e).__name__}: {e}'}

    def run(self):
        while not self.stop_flag:
            with self.lock:
                item = self.pending
                self.pending = None
            if item is None:
                time.sleep(0.005)
                continue
            seq, jpg = item
            try:
                t0 = time.perf_counter()
                s = socket.create_connection((self.host, self.port), timeout=180)
                s.settimeout(180)
                s.sendall(struct.pack('!III', SEND_W, 0, len(jpg)) + jpg)
                n = struct.unpack('!I', _recv_exact(s, 4))[0]
                res = json.loads(_recv_exact(s, n).decode('utf-8'))
                s.close()
                self.rtt_ms = (time.perf_counter() - t0) * 1000
                self.connected = True
                self.last_error = ''
                if res.get('ok'):
                    self.frames += 1
                    with self.lock:
                        self.result = res
                        self.done_seq = seq
                    if self.on_result:
                        self.on_result(res)
                else:
                    self.last_error = res.get('error', '未知错误')
            except Exception as e:
                self.connected = False
                self.last_error = f'{type(e).__name__}: {e}'
                time.sleep(0.6)


def _recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            raise ConnectionError('连接中断')
        buf += c
    return buf


def check_board(host, port, sys_port=9090, tries=6, delay=3.0):
    """带重试的板卡连接检查; 失败时给出明确诊断。
       返回 (是否成功, 心跳响应, 诊断文字)"""
    last = ''
    for i in range(tries):
        try:
            s = socket.create_connection((host, port), timeout=8)
            payload = json.dumps({'mode': 'ocr'}).encode()
            # 空帧心跳(不改变模式)
            s.sendall(struct.pack('!III', 0, 0, 0))
            n = struct.unpack('!I', _recv_exact(s, 4))[0]
            res = json.loads(_recv_exact(s, n).decode('utf-8'))
            s.close()
            return True, res, ''
        except ConnectionRefusedError:
            last = 'refused'
        except (socket.timeout, TimeoutError):
            last = 'timeout'
        except Exception as e:
            last = f'{type(e).__name__}'
        if i < tries - 1:
            print(f"  ⏳ 第 {i+1}/{tries} 次连接失败({last})，"
                  f"{delay:.0f} 秒后重试（服务可能正在初始化，需约 20 秒）...")
            time.sleep(delay)

    # ---- 诊断 ----
    diag = []
    sys_ok = False
    try:
        s2 = socket.create_connection((host, sys_port), timeout=6)
        s2.close()
        sys_ok = True
    except Exception:
        pass
    if sys_ok:
        diag.append(f"板卡系统在线（{host}:{sys_port} 可连），但识别服务没在跑 ✗")
        diag.append("解决：在板卡 Jupyter 里打开「启动识别服务.ipynb」，点第 ① 格运行；")
        diag.append("      等看到「监听 0.0.0.0:9999 等待 PC 连接」再重试本程序 ✓")
    else:
        diag.append(f"连不上板卡（{host}:{sys_port} 也不通）✗")
        diag.append("解决：① 板卡电源灯亮?  ② 网线插好(两端)?  ③ 在 cmd 里运行 ipconfig")
        diag.append("      本机应有一个 IPv4 是 192.168.2.1")
    return False, None, '\n     '.join(diag)


# ============================================================
#  画面叠加显示
# ============================================================
def log_result(res):
    """把识别结果追加写入 UTF-8 日志(避免控制台显示不了中文)"""
    if not res:
        return
    try:
        with open('result_log.txt', 'a', encoding='utf-8') as f:
            ts = time.strftime('%H:%M:%S')
            bars = ' | '.join(f"{b['type']}:{b['data']}" for b in res.get('bars', []))
            if res.get('isbn'):
                bars += f"  (ISBN-13: {res['isbn']})"
            txt = ' / '.join(res.get('texts', []))
            tm = res.get('timing', {})
            f.write(f"[{ts}] 条码: {bars or '(无)'}\n")
            if txt:
                f.write(f"         文字: {txt}\n")
            f.write(f"         耗时: PL {tm.get('pl')}ms 条码 {tm.get('t_bar')}ms "
                    f"OCR {tm.get('t_ocr')}ms 合计 {tm.get('total')}ms\n")
    except Exception:
        pass


def normalize_illum(img):
    """抑制光照不均/暗角/阴影 —— 只作用于"发给板卡的图", 不影响本地显示。
       摄像头画面经此处理后, 板卡上的全局阈值二值化质量会好很多,
       从而大幅减少"把暗角/暗物体当成文字"的误检。"""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        bg = cv2.GaussianBlur(small, (0, 0), 12)
        bg = cv2.resize(bg, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
        norm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(norm)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    except Exception:
        return img


def start_console_reader(q):
    """后台线程读取控制台输入 —— 即使 OpenCV 窗口没获得焦点/被输入法拦截,
       也能在控制台输入命令(回车执行)来控制程序。"""
    def _loop():
        try:
            for line in sys.stdin:
                c = (line or '').strip().lower()
                if c:
                    q.append(c[0])
        except Exception:
            pass
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def get_screen_size():
    """获取屏幕分辨率(Windows)"""
    try:
        import ctypes
        u = ctypes.windll.user32
        return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def fit_to_screen(frame, max_w, max_h):
    """把画面等比缩到屏幕内(只影响显示, 不影响发给板卡的分辨率)"""
    h, w = frame.shape[:2]
    if w <= max_w and h <= max_h:
        return frame, 1.0
    s = min(max_w / float(w), max_h / float(h))
    return cv2.resize(frame, (max(2, int(w * s)), max(2, int(h * s))),
                      interpolation=cv2.INTER_AREA), s


def open_camera(idx, tries=10, verbose=True):
    """打开摄像头: 尝试多种后端 + 预热若干帧(有些摄像头前几帧读不到)。
       返回 (cap 或 None, 错误说明)"""
    backends = []
    if os.name == 'nt':
        backends = [('DSHOW', cv2.CAP_DSHOW), ('MSMF', cv2.CAP_MSMF), ('ANY', cv2.CAP_ANY)]
    else:
        backends = [('ANY', cv2.CAP_ANY)]
    last_err = ''
    for name, be in backends:
        cap = None
        try:
            cap = cv2.VideoCapture(idx, be)
        except Exception as e:
            last_err = f'{name}: {type(e).__name__}'
            continue
        if not cap.isOpened():
            last_err = f'{name}: 打不开'
            try:
                cap.release()
            except Exception:
                pass
            continue
        # 预热：连续尝试读帧
        ok = False
        for i in range(tries):
            try:
                ok, f = cap.read()
            except Exception:
                ok, f = False, None
            if ok and f is not None:
                break
            time.sleep(0.15)
        if ok:
            if verbose:
                print(f"  摄像头打开成功（后端 {name}，预热 {i+1} 次）")
            return cap, ''
        last_err = f'{name}: 打开成功但读不到画面'
        try:
            cap.release()
        except Exception:
            pass
    return None, last_err


def probe_cameras(maxidx=4):
    """列出可用的摄像头编号"""
    found = []
    for i in range(maxidx):
        cap, _ = open_camera(i, tries=2, verbose=False)
        if cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, w, h))
            cap.release()
    return found


class VoteBox:
    """多帧投票: 把连续多帧的识别结果按【框的位置】投票, 形近字错误会互相纠正。

    关键: 用"框的量化位置"作为投票键(而不是框序号) —— 手持摄像头时框会移动,
    用序号会导致"显示的文字"和"画面上的框"对不上。
       例: 中华人民其和加 / 中华人民共和和太 / 中华人民共和国 -> 中华人民共和国"""

    GRID = 90.0          # 位置量化格(像素), 越大越容忍手持抖动

    def __init__(self, keep=10):
        self.keep = keep
        self.hist = {}          # 位置键 -> [最近若干帧的文本]

    @classmethod
    def key_of(cls, box):
        if not box:
            return None
        x, y, w, h = box
        cx = (x + w / 2.0) / cls.GRID
        cy = (y + h / 2.0) / cls.GRID
        return (int(cx), int(cy))

    def add(self, res):
        rt = (res or {}).get('roi_texts') or []
        rois = (res or {}).get('rois') or []
        for i, lines in enumerate(rt):
            if not lines:
                continue
            txt = (lines[0] or '').strip()
            if not txt:
                continue
            k = self.key_of(rois[i] if i < len(rois) else None)
            if k is None:
                k = ('idx', i)
            h = self.hist.setdefault(k, [])
            h.append(txt)
            if len(h) > self.keep:
                h.pop(0)

    def best_for_box(self, box, idx=0):
        """按当前框的位置取投票结果"""
        k = self.key_of(box)
        if k is not None and k in self.hist:
            return self._vote(self.hist[k])
        return self._vote(self.hist.get(('idx', idx)) or [])

    def _vote(self, hist):
        if not hist:
            return '', 0
        cnt = {}
        for t in hist:
            cnt[t] = cnt.get(t, 0) + 1
        top, topn = '', 0
        for k, v in cnt.items():
            if v > topn:
                top, topn = k, v
        if len(top) >= 2:
            same = [t for t in hist if len(t) == len(top)]
            if len(set(same)) > 1:
                chars = []
                for pos in range(len(top)):
                    cc = {}
                    for t in same:
                        c = t[pos]
                        cc[c] = cc.get(c, 0) + 1
                    bc, bv = '', 0
                    for c, v in cc.items():
                        if v > bv:
                            bc, bv = c, v
                    chars.append(bc)
                cons = ''.join(chars)
                if cons:
                    return cons, len(hist)
        return top, len(hist)

    def best(self, idx):
        return self._vote(self.hist.get(('idx', idx)) or [])


def draw_overlay(frame, res, stats, votes=None):
    h, w = frame.shape[:2]
    # 坐标基准: 板卡实际处理的尺寸(可能因超限被缩放), 而非发送尺寸
    base = None
    if res:
        base = res.get('proc_size') or res.get('size')
    sx = (w / float(base[0])) if base else 1.0
    sy = (h / float(base[1])) if base else 1.0

    if res:
        # 坐标偏移: 若板卡做了裁剪, 框要挪回原图位置
        ox, oy = ((res.get('crop') or [0, 0, 0, 0])[:2])
        rt_all = res.get('roi_texts') or []
        # 文字区域框(蓝色) + 编号徽标 —— 只画"确实识别出文字"的框, 避免多余框干扰
        for idx, r in enumerate(res.get('rois', []), 1):
            has_txt = bool(rt_all[idx - 1]) if idx - 1 < len(rt_all) else False
            voted = ''
            if votes is not None:
                vt, _vn = votes.best_for_box(r, idx - 1)
                voted = vt
            if not has_txt and not voted:
                continue                      # 没识别出文字 -> 不画框(减少干扰)
            x, y, rw, rh = r
            x = int((x + ox) * sx); y = int((y + oy) * sy)
            rw, rh = int(rw * sx), int(rh * sy)
            cv2.rectangle(frame, (x, y), (x + rw, y + rh), (255, 160, 0), 2)
            ly = max(0, y - 26)
            cv2.rectangle(frame, (x, ly), (x + 28, ly + 24), (255, 160, 0), -1)
            frame = put_cn(frame, str(idx), (x + 7, ly + 1), (255, 255, 255), 20)
        # 条码框(绿)
        for b in res.get('bars', []):
            x, y, bw, bh = b['rect']
            x, y = int((x + ox) * sx), int((y + oy) * sy)
            bw, bh = int(bw * sx), int(bh * sy)
            cv2.rectangle(frame, (x - 6, y - 6), (x + bw + 6, y + bh + 6), (0, 200, 0), 2)
            cv2.putText(frame, b['type'], (x - 6, y - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 200, 0), 2)

    # 结果面板(按框编号显示, 与画面上的蓝色徽标对应)
    lines = []
    if res:
        for b in res.get('bars', [])[:3]:
            lines.append(f"条码 {b['type']}: {b['data']}")
        if res.get('isbn'):
            lines.append(f"ISBN-13: {res['isbn']}")
        rt = res.get('roi_texts') or []
        rois_now = res.get('rois') or []
        if rt:
            for i, ls in enumerate(rt):
                box = rois_now[i] if i < len(rois_now) else None
                shown = ''
                if votes is not None:
                    vt, vn = votes.best_for_box(box, i)
                    if vt:
                        tag = f"({vn}票)" if vn > 1 else ""
                        shown = f"[{i+1}] {vt} {tag}".rstrip()
                if not shown:
                    if ls:
                        parts = [f"[{i+1}] {ls[0]}"] + [f"     {t}" for t in ls[1:]]
                        lines.extend(parts)
                    continue          # 没文字的行不显示, 减少刷屏
                lines.append(shown)
        else:
            for t in res.get('texts', [])[:6]:
                lines.append(t)
        tm = res.get('timing', {})
        lines.append(f"PL {tm.get('pl')}ms | barcode {tm.get('t_bar')}ms | OCR {tm.get('t_ocr')}ms"
                     f" | total {tm.get('total')}ms")
    if not lines:
        lines = ['(等待识别结果 / waiting ...)']
    lines = lines[:11]        # 面板别占太多画面

    pad, lh = 10, 28
    bh_ = pad * 2 + lh * (len(lines) + 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bh_), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    y = h - bh_ + pad + 6
    for i, ln in enumerate(lines):
        is_last = (i == len(lines) - 1)
        color = (160, 160, 160) if is_last else (0, 255, 255)
        # 用 PIL 画(支持中文), 每行单独转换
        frame = put_cn(frame, ln[:60], (pad, y), color, 20 if is_last else 22)
        y += lh
    # 状态行
    conn, rtt, nf, err, mode, backend = stats
    st = (f"模式={mode} 后端={backend} | {'已连接' if conn else '未连接'} "
          f"| 往返 {rtt:.0f}ms | 已识别 {nf} 帧")
    cv2.putText(frame, st, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 0) if conn else (0, 0, 255), 1)
    if err and not conn:
        cv2.putText(frame, f"错误: {err[:70]}", (pad, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
    # 按键提示
    cv2.putText(frame, "[q]quit [b]barcode [o]ocr [a]acc [g]gate [t]strict [i]illum [f]fit [s]shot"
                       "   (click this window first!)",
                (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


# ============================================================
def main():
    global SEND_W
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=HOST)
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--cam', type=int, default=0, help='摄像头编号')
    ap.add_argument('--source', default='', help='视频文件(替代摄像头, 测试用)')
    ap.add_argument('--width', type=int, default=1280, help='发送宽度(≤1920)')
    ap.add_argument('--test', type=int, default=0, help='无界面测试模式: 只发 N 帧')
    ap.add_argument('--mode', default='ocr', choices=['barcode', 'ocr', 'ocr_accurate'])
    ap.add_argument('--fps', type=float, default=0, help='限制发送帧率(0=尽快)')
    ap.add_argument('--crop', default='', help='只识别该区域, 格式 x0,y0,x1,y1 (提升文字识别效果)')
    ap.add_argument('--no-illum', action='store_true', help='关闭光照归一化(默认开启)')
    a = ap.parse_args()

    SEND_W = a.width
    illum = False        # 光照归一化默认关闭(合成场景实测会多出误检; 需要时按 i 开启)
    # ---- 显示尺寸: 自动适配屏幕, 保证底部结果面板可见 ----
    SW, SH = get_screen_size()
    DISP_W, DISP_H = int(SW * 0.92), int(SH * 0.80)
    disp_full = False
    print(f"  屏幕 {SW}x{SH} → 显示上限 {DISP_W}x{DISP_H}（按 f 切换原始尺寸）")
    crop = None
    if a.crop:
        try:
            crop = [int(v) for v in a.crop.replace('，', ',').split(',')]
            assert len(crop) == 4
        except Exception:
            print(f"  ⚠️ --crop 格式不对: {a.crop}  (应形如 0,860,700,1070)")
            crop = None

    print("=" * 66)
    print("  PC 端实时识别客户端")
    print("=" * 66)
    link = BoardLink(a.host, a.port)
    link.start()
    print(f"  板卡: {a.host}:{a.port}")
    okb, hb, diag = check_board(a.host, a.port)
    if okb:
        print(f"  ✅ 板卡已连接 (版本 {hb.get('version', '?')}, 服务模式 {hb.get('mode')})")
        r = link.send_cmd({'mode': a.mode})
        if r.get('ok'):
            print(f"  已设置为: {a.mode} 模式")
    else:
        print(f"  ❌ 板卡连接失败")
        print(f"     {diag}")
        if a.test:
            return
        print("\n  （仍将显示摄像头画面，但不会有识别结果；")
        print("    解决后按 Ctrl+C 退出并重新运行本程序）")

    # 明确设置裁剪(即使为空也要发, 避免上一次会话的裁剪残留)
    r = link.send_cmd({'crop': crop})
    if crop:
        print(f"  裁剪区域: {crop}  ({r.get('ok')})")
    else:
        print(f"  裁剪区域: 无(全画面)  ({r.get('ok')})")
    # ---- 视频源 ----
    static_img = None
    if a.source:
        ext = os.path.splitext(a.source)[1].lower()
        if ext in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
            static_img = cv2.imread(a.source)
            if static_img is None:
                print(f"  ❌ 读不到图片: {a.source}"); return
            print(f"  静态测试源: {a.source}  {static_img.shape[1]}x{static_img.shape[0]}")
            cap = None
        else:
            cap = cv2.VideoCapture(a.source)
            static_img = None
            print(f"  视频源(测试): {a.source}")
    else:
        cap, cerr = open_camera(a.cam)
        static_img = None
        if cap is None:
            # ⭐ 自动兜底: 指定编号打不开时, 自动换其他编号(免去手动选摄像头)
            print(f"  ⚠️ 摄像头 {a.cam} 打不开, 自动尝试其他编号 ...")
            for i in range(4):
                if i == a.cam:
                    continue
                c2, _e = open_camera(i, tries=4, verbose=False)
                if c2 is not None:
                    cap, a.cam = c2, i
                    print(f"  ✅ 自动改用摄像头编号 {i}")
                    try:
                        with open('cam.txt', 'w', encoding='utf-8') as f:
                            f.write(str(i))
                        print(f"     已记住(cam.txt = {i}), 下次直接用这个 ✓")
                    except Exception:
                        pass
                    break
        if cap is None:
            print(f"\n  ❌ 所有摄像头都打不开/读不到画面  ({cerr})")
            print("     可能原因：")
            print("       ① 摄像头没插好 → 换个 USB 口试试")
            print("       ② 被其他程序占用 → 关掉：Windows「相机」App、微信、QQ、"
                  "以及之前运行的本程序窗口")
            print("       ③ 摄像头驱动异常 → 在「设备管理器」看看有没有黄色感叹号")
            try:
                input("\n     按回车键关闭 ...")
            except EOFError:
                pass
            return
        print(f"  摄像头: index {a.cam}")
    if cap is not None and not cap.isOpened():
        print("  ❌ 打不开视频源(检查摄像头或文件路径)")
        return
    if cap is not None:
        ok, _f = cap.read()
        if ok:
            print(f"  画面尺寸: {_f.shape[1]}x{_f.shape[0]}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # 视频文件: 按原始帧率播放(否则会飞快播完)
        if a.source and a.fps <= 0:
            vf = cap.get(cv2.CAP_PROP_FPS) or 0
            if 1 < vf < 120:
                a.fps = vf
                print(f"  播放帧率: {vf:.1f} FPS (按视频原速)")

    mode = a.mode
    gate = True
    strict = True
    cmdq = []                                  # 控制台命令队列
    votes = VoteBox()
    start_console_reader(cmdq)
    print("\n  ⚠️ 按键无效时: ① 先用鼠标点一下画面窗口 ② 把输入法切到英文(Shift)")
    print("     也可以直接在【控制台】里输入命令后回车: q/o/b/a/g/t/i/f/s/r")
    n_sent, n_recv = 0, 0
    n_fail, n_log = 0, -1
    _topmost_done = False
    t_start = time.time()
    last_send = 0.0
    fps_disp = 0.0
    t_fps = time.time()
    test_timeout = 240.0        # 测试模式最长等待(秒), 超时退出

    print("  按键: q退出 b只条码 o文字 a高精度 g门控 r重载 s截图")
    print("-" * 66)

    try:
        while True:
            if static_img is not None:
                frame = static_img.copy()
                ok = True
            else:
                ok, frame = cap.read()
            if not ok:
                if a.source:            # 视频播完 → 重头再来
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        break
                else:
                    # 摄像头: 单次读失败不退出, 连续失败才退出并提示
                    n_fail += 1
                    if n_fail >= 60:
                        print("\n  ❌ 摄像头连续读帧失败, 可能在运行中被其他程序抢占了")
                        print("     请关掉可能占用摄像头的程序后重试")
                        try:
                            input("     按回车键关闭 ...")
                        except EOFError:
                            pass
                        break
                    time.sleep(0.05)
                    continue
            n_fail = 0
            # 缩放到发送宽度
            if frame.shape[1] != SEND_W:
                s = SEND_W / frame.shape[1]
                frame = cv2.resize(frame, (SEND_W, int(frame.shape[0] * s)))

            res, conn, rtt, nf, err = link.snapshot()

            # ---- 发送(不阻塞显示) ----
            now = time.time()
            if (a.fps <= 0) or (now - last_send >= 1.0 / a.fps):
                # 光照归一化: 抑制暗角/阴影(只影响发给板卡的图)
                send_frame = normalize_illum(frame) if illum else frame
                ok2, buf = cv2.imencode('.jpg', send_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if ok2:
                    link.submit(buf.tobytes())
                    n_sent += 1
                    last_send = now

            # ---- 显示(自动适配屏幕, 保证底部结果面板可见) ----
            if a.test:
                if res and nf > n_recv:
                    n_recv = nf
                    log_result(res)          # 写入 UTF-8 日志(控制台显示不了中文时看这里)
                    tm = res.get('timing', {})
                    print(f"  帧{nf}: 条码={[b['data'] for b in res['bars']]} "
                          f"文字={res['texts'][:2]} PL={tm.get('pl')}ms 合计={tm.get('total')}ms")
                if nf >= a.test:
                    break
                if time.time() - t_start > test_timeout:
                    print(f"  ⏱ 测试超时({test_timeout:.0f}s): 已发 {n_sent} 帧, "
                          f"完成识别 {nf} 帧")
                    if not link.connected:
                        print(f"     板卡未连接: {link.last_error}")
                    break
                time.sleep(0.02)
                continue

            if disp_full:
                disp_frame, _sc = frame, 1.0
            else:
                disp_frame, _sc = fit_to_screen(frame, DISP_W, DISP_H)
            if res and nf > n_log:
                n_log = nf
                votes.add(res)           # 多帧投票
                log_result(res)          # 每帧结果写入 UTF-8 日志

            disp = draw_overlay(disp_frame.copy(), res,
                                (conn, rtt, nf, err, res.get('mode', mode) if res else mode,
                                 res.get('backend', '-') if res else '-'),
                                votes=votes)
            # 显示帧率
            if time.time() - t_fps >= 0.5:
                fps_disp = n_sent / max(time.time() - t_start, 0.001)
                t_fps = time.time()
            cv2.putText(disp, f"发送 {fps_disp:.1f} FPS", (disp.shape[1] - 190, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.imshow('PYNQ-Z2 Real-time Recognition', disp)
            # 窗口置顶, 减少"键没反应"的情况(焦点被别的窗口抢走)
            if not _topmost_done:
                try:
                    cv2.setWindowProperty('PYNQ-Z2 Real-time Recognition',
                                          cv2.WND_PROP_TOPMOST, 1)
                except Exception:
                    pass
                _topmost_done = True
            k = cv2.waitKey(1) & 0xFF
            if cmdq:                       # 控制台输入的命令优先
                k = ord(cmdq.pop(0))
            if k == ord('q'):
                break
            elif k in (ord('b'), ord('o'), ord('a')):
                mode = {'b': 'barcode', 'o': 'ocr', 'a': 'ocr_accurate'}[chr(k)]
                r = link.send_cmd({'mode': mode})
                print(f"  → 模式: {mode}  ({r.get('ok')})")
            elif k == ord('g'):
                gate = not gate
                link.send_cmd({'gate': gate})
                print(f"  → 门控: {'开' if gate else '关'}")
            elif k == ord('r'):
                link.send_cmd({'reset': True})
                print("  → 已请求板卡重载硬件")
            elif k == ord('s'):
                fn = f"shot_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn, disp)
                print(f"  → 截图已保存: {fn}")
            elif k == ord('i'):
                illum = not illum
                print(f"  → 光照归一化: {'开(抑制暗角/阴影)' if illum else '关(原始画面)'}")
            elif k == ord('t'):
                strict = not strict
                r2 = link.send_cmd({'strict': strict})
                print(f"  → 严格文字校验: {'开(滤掉非文字物体)' if strict else '关(更不容易漏文字)'}"
                      f"  ({r2.get('ok')})")
            elif k == ord('f'):
                disp_full = not disp_full
                print(f"  → 显示模式: {'原始尺寸' if disp_full else '适配屏幕'}")
            elif k in (ord('+'), ord('=')):
                DISP_H = int(DISP_H * 1.1)
                print(f"  → 显示放大: 上限 {DISP_W}x{DISP_H}")
            elif k in (ord('-'), ord('_')):
                DISP_H = int(DISP_H * 0.9)
                print(f"  → 显示缩小: 上限 {DISP_W}x{DISP_H}")
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        link.stop_flag = True
        el = time.time() - t_start
        print("-" * 66)
        print(f"  共发送 {n_sent} 帧, 完成识别 {nf} 帧, 用时 {el:.1f}s "
              f"→ 端到端 {nf/max(el,0.001):.2f} FPS")


if __name__ == '__main__':
    main()
