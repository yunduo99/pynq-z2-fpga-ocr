# -*- coding: utf-8 -*-
"""
PYNQ-Z2 识别终端 —— 公共模块
============================================================
把「硬件加速流水线 + 条码解码 + 文字OCR」封装成函数, Notebook 里只写一行调用。

支持三种场景:
    单张图片   →  po.run_image('photo.jpg')
    视频       →  po.run_video('video.mp4', step=4, maxframes=8)
    批量图片   →  po.run_folder('docs', max_seconds=120)

设计要点(都是实测踩坑后的结论):
  * 硬件与 OCR 引擎是全局单例, 首次调用自动初始化, 之后复用(省去 3~4 秒)
  * 每次运行前有"运行锁", 防止两个单元格同时跑(并发会互相拖慢 13 倍)
  * 变化门控(GATE): 画面未变化时跳过 OCR, 实测快 14.8 倍
  * 语言: chi_sim(中文, 默认) / eng(英文, 快 4 倍)  —— ⚠️ 不要用 chi_sim+eng(中文会变拉丁乱码)
  * PL 处理分辨率上限 1920x1080(形态学 IP 行缓存 MAX_W=1920)
  * 所有等待都有超时保护, 不会永久卡死
"""
import os, sys, time, glob, ctypes, re
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import cv2

OUTDIR = '/home/xilinx/jupyter_notebooks/pynq_project'
VERSION = 'v9-dual-ps-pc-option-2026.11.02'    # 更新后重启服务/Kernel
BITFILE = OUTDIR + '/pynq_dma35.bit'
TESSDATA_STD = '/usr/share/tesseract-ocr/5/tessdata'
TESSDATA_FAST = OUTDIR + '/tessdata_fast'
LOCKFILE = OUTDIR + '/.pnq_run.lock'

MAXW, MAXH = 1920, 1080          # PL 分辨率上限
ROTMAP = {'0': None, '90': cv2.ROTATE_90_COUNTERCLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE,
          '270': cv2.ROTATE_90_CLOCKWISE, 'cw': cv2.ROTATE_90_CLOCKWISE, '180': cv2.ROTATE_180}
ROTNAME = {None: '不旋转', cv2.ROTATE_90_COUNTERCLOCKWISE: '逆时针90',
           cv2.ROTATE_90_CLOCKWISE: '顺时针90', cv2.ROTATE_180: '180'}
CTRL, ARG0, REG_W, REG_H, REG_EN = 0x00, 0x10, 0x18, 0x20, 0x28
MM2S_DMACR, MM2S_DMASR, MM2S_SA, MM2S_LENGTH = 0x00, 0x04, 0x18, 0x28
S2MM_DMACR, S2MM_DMASR, S2MM_DA, S2MM_LENGTH = 0x30, 0x34, 0x48, 0x58

# 只解码常用条码类型(跳过其余符号, 实测省 20% 时间)
try:
    from pyzbar.pyzbar import ZBarSymbol
    SYMS = [ZBarSymbol.QRCODE, ZBarSymbol.EAN13, ZBarSymbol.EAN8, ZBarSymbol.CODE128,
            ZBarSymbol.CODE39, ZBarSymbol.UPCA, ZBarSymbol.UPCE, ZBarSymbol.I25]
except Exception:
    SYMS = None


def _log(*a):
    print(*a, flush=True)


# ============================================================
#  图像增强工具(提高识别率)
# ============================================================
def add_border(img, px=16, val=255):
    """加白边: tesseract 对带白边的文本行识别率明显更高"""
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=val)


def clahe(gray, clip=2.5, grid=8):
    """限制对比度自适应直方图均衡: 改善光照不均"""
    try:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray)
    except Exception:
        return gray


def unsharp(gray, k=5, amount=1.2):
    """锐化(USM): 改善轻微模糊"""
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def upscale_to_height(img, target_h=36, max_scale=3.0):
    """按目标字高放大(小字放大后识别率显著提高)"""
    h = img.shape[0]
    if h <= 0:
        return img, 1.0
    s = min(max_scale, max(1.0, target_h / float(h)))
    if s <= 1.01:
        return img, 1.0
    return cv2.resize(img, (int(img.shape[1] * s), int(h * s)), interpolation=cv2.INTER_CUBIC), s


def deskew(img, max_angle=8.0):
    """倾斜校正(小幅旋转)。只在估计可靠时才旋转(小裁剪块上容易估错, 故有严格门槛)"""
    try:
        h, w = img.shape
        if h < 24 or w < 60:
            return img, 0.0
        inv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(inv)
        if coords is None or len(coords) < 200:
            return img, 0.0
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(coords)
        if rw < rh:
            ang += 90
        if ang > 45:
            ang -= 90
        # 只有"细长条"(像文本行) 且角度很小时才认为估计可靠
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side <= 0 or long_side / short_side < 3.0:
            return img, 0.0
        if abs(ang) < 0.8 or abs(ang) > max_angle:
            return img, 0.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), ang
    except Exception:
        return img, 0.0


def prep_for_ocr(img, target_h=36, do_deskew=False, border=16, verbose=False):
    """OCR 前的统一预处理: 放大 -> (可选)倾斜校正 -> 加白边
       注意: 默认不做倾斜校正 —— 实测在小裁剪块上估角不可靠, 反而降低识别率"""
    notes = []
    out, s = upscale_to_height(img, target_h)
    if s > 1.01:
        notes.append(f"放大{s:.1f}x")
    if do_deskew:
        out, ang = deskew(out)
        if abs(ang) > 0.8:
            notes.append(f"校正{ang:.1f}度")
    if border:
        out = add_border(out, border)
        notes.append(f"白边{border}")
    return out, '+'.join(notes) if notes else '原图'


def _is_barcode_noise(text):
    """判断是否为"条码下方那串数字"的识别噪声:
       无中文 + 数字很多 + 不含网址字符(./:@)  -> 视为噪声(条码内容已由解码给出)"""
    s = text.strip()
    if not s:
        return False
    if any('\u4e00' <= c <= '\u9fff' for c in s):
        return False
    if any(c in './:@' for c in s):
        return False
    digits = sum(1 for c in s if c.isdigit())
    return digits >= 5


def fix_common_confusions(text):
    import re
    t = text
    # 纯数字串里的字母误认
    def fix_num(m):
        s = m.group(0)
        if re.search(r'\d{3,}', s):
            return s.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        return s
    t = re.sub(r'[0-9OoIl\-\s]{4,}', fix_num, t)
    # 中文文本里的破折号统一
    t = t.replace('—', '-').replace('–', '-')
    return t


def text_score(lines):
    """给一组识别结果打分(替代不可靠的 tesseract 置信度 API)
       规则: 有效字符(CJK/字母数字) 加分, 符号 扣分, 行数 小幅加分"""
    if not lines:
        return -1.0
    s = ''.join(lines)
    if not s:
        return -1.0
    good = sum(1 for c in s if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    bad = len(s) - good
    return good - 2.0 * bad + 2.0 * len(lines)


def is_garbage(text, min_alnum=2, max_symbol_ratio=0.35):
    """判断一行是不是乱码(过滤掉 tesseract 的无效输出)"""
    s = clean_line(text)
    if not s:
        return True
    alnum = sum(1 for c in s if c.isalnum() or '\u4e00' <= c <= '\u9fff')
    if alnum < min_alnum:
        return True
    # 短于 3 字符且不含数字 -> 基本是碎片(如 'ER'、'全一')
    if len(s) < 3 and not any(c.isdigit() for c in s):
        return True
    if len(s) >= 4 and (len(s) - alnum) / len(s) > max_symbol_ratio:
        return True
    return False


def clean_line(text):
    """去掉识别结果两端粘连的孤立符号(引号/破折号/点号等)"""
    s = (text or '').strip()
    return s.strip(' "\'“”‘’·-—–.,、;:!?|/\\_=+*#~`^<>[]{}()')


def looks_like_noise(text):
    """识别拉丁字母噪声行(如纹理/插图被 tesseract 编造的内容):
       条件: 无中文 + 无网址字符 + 词数 ≤3 + 某个词出现 ≥4 个连续辅音"""
    s = (text or '').strip()
    if not s or any('\u4e00' <= c <= '\u9fff' for c in s):
        return False
    if ('://' in s) or ('@' in s) or ('.' in s and '/' in s) or s.count('.') >= 2:
        return False                       # 真正的网址/邮箱才保留(单个冒号不算)
    words = [w for w in re.split(r'[^A-Za-z]+', s) if w]
    if not words or len(words) > 3:
        return False                      # 词多更像正常英文句子
    for w in words:
        run = 0
        for ch in w.lower():
            if ch in 'aeiou':
                run = 0
            else:
                run += 1
                if run >= 4 and len(w) >= 5:
                    return True
    return False


def is_latin_only(text):
    s = (text or '')
    return (not any('\u4e00' <= c <= '\u9fff' for c in s)) and any(c.isalpha() for c in s)


def is_url_like(text):
    s = (text or '')
    return ('@' in s) or ('://' in s) or ('.' in s and '/' in s) or s.count('.') >= 2


def merge_lines(lines, limit=12, drop_latin=False):
    """合并多来源的识别行: 清洗 -> 去乱码 -> 去重(忽略空格/大小写) -> 去掉互相包含的碎片"""
    out, seen = [], set()
    for l in lines:
        c = fix_common_confusions(clean_line(l))
        if not c:
            continue
        key = re.sub(r'\s+', '', c).lower()      # 忽略空格/大小写做去重
        if key in seen or is_garbage(c):
            continue
        seen.add(key)
        out.append(c)
    # 若某行被另一行完整包含, 且自身很短, 视为碎片丢弃
    final = []
    for c in out:
        if len(c) <= 4 and any(c != o and c in o for o in out):
            continue
        final.append(c)
    # 相似行合并(分块重叠区常会出现同一内容的多种识别结果)
    try:
        import difflib
        merged = []
        for c in final:
            hit = None
            for i, k in enumerate(merged):
                if difflib.SequenceMatcher(None, c, k).ratio() >= 0.75:
                    hit = i
                    break
            if hit is None:
                merged.append(c)
            elif text_score([c]) > text_score([merged[hit]]):
                merged[hit] = c            # 保留内容更完整的那条
        final = merged
    except Exception:
        pass
    # 拉丁字母噪声行(纹理/插图编造的)
    final = [c for c in final if not looks_like_noise(c)]
    # 可选: 丢掉所有"纯拉丁"行(中文场景可开启, 会保留网址)
    if drop_latin:
        final = [c for c in final if (not is_latin_only(c)) or is_url_like(c)]
    return final[:limit]


# ============================================================
#  运行锁: 防止两个单元格并发(板卡单核, 并发会拖慢十几倍)
# ============================================================
def _acquire_lock():
    if os.path.exists(LOCKFILE):
        try:
            pid = int(open(LOCKFILE).read().strip())
            if os.path.exists(f'/proc/{pid}'):
                _log(f"⚠️ 已有任务在运行 (PID {pid})。请等它结束, 或 Kernel → Restart 后再试。")
                return False
        except Exception:
            pass
    try:
        with open(LOCKFILE, 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass
    return True


def _release_lock():
    try:
        os.remove(LOCKFILE)
    except Exception:
        pass


# ============================================================
#  终端类
# ============================================================
class Terminal:
    def __init__(self):
        self.ready = False
        self.ocr = None
        self.ocr_lib = None
        self.plw = self.plh = 0
        self.images = 0
        self.total_ms = 0.0

    # ---------------- 初始化 ----------------
    def init(self, bit=BITFILE, ocr_lang='chi_sim', morph_mode=1,
             tessdata=None, force=False):
        if self.ready and not force:
            return self
        if force:
            from performance_runtime import close_runtime
            close_runtime(self)
            if self.ocr and self.ocr_lib:
                self.ocr_lib.TessBaseAPIDelete(self.ocr)
                self.ocr = None
            self.ready = False
        from pynq import Overlay, allocate, MMIO

        _log("=== 初始化硬件 (首次会慢 3~5 秒) ===")
        t0 = time.perf_counter()
        self.ol = Overlay(bit)
        self.bin_mmio = MMIO(self.ol.ip_dict['binarize_0']['phys_addr'], 0x1000)
        self.mor_mmio = MMIO(self.ol.ip_dict['morphology_0']['phys_addr'], 0x1000)
        self.dmmio = self.ol.axi_dma_0.mmio
        # 缓冲区按 PL 上限一次性分配(2M 像素 x 4B x 2 = 16.6MB)
        NMAX = MAXW * MAXH
        self.inbuf = allocate(shape=(NMAX,), dtype=np.uint32)
        self.outbuf = allocate(shape=(NMAX,), dtype=np.uint32)
        self.LUT = np.array([(v << 16) | (v << 8) | v for v in range(256)], dtype=np.uint32)
        self.morph_mode = morph_mode
        _log(f"  Overlay: {os.path.basename(bit)}  ({(time.perf_counter()-t0):.1f}s)")

        # Prefer the existing fast model, fail visibly if models are absent.
        if tessdata is None:
            candidates = [os.environ.get('PYNQ_TESSDATA'), TESSDATA_FAST,
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tessdata'),
                          TESSDATA_STD, '/usr/share/tesseract-ocr/4.00/tessdata']
            languages = ocr_lang.split('+')
            tessdata = next((d for d in candidates if d and all(
                os.path.isfile(os.path.join(d, lang + '.traineddata')) for lang in languages)), None)
        if not tessdata:
            raise RuntimeError('未找到 OCR 模型，请设置 PYNQ_TESSDATA 或提供 tessdata_fast')
        from ocr_pipeline import bind_api
        lib = bind_api(ctypes.CDLL(os.environ.get('PYNQ_TESSERACT_LIB',
                            '/usr/lib/arm-linux-gnueabihf/libtesseract.so.4')))
        t0 = time.perf_counter()
        self.ocr_lib = lib
        self.ocr = lib.TessBaseAPICreate()
        if not self.ocr:
            raise RuntimeError('Tesseract 初始化句柄失败')
        rc = lib.TessBaseAPIInit3(self.ocr, tessdata.encode(), ocr_lang.encode())
        if rc != 0:
            lib.TessBaseAPIDelete(self.ocr)
            self.ocr = None
            raise RuntimeError(f'Tesseract 模型加载失败 rc={rc}: {tessdata} / {ocr_lang}')
        self._smart_cache = None
        lib.TessBaseAPISetPageSegMode(self.ocr, 6)
        for var, val in ((b'load_system_dawg', b'0'), (b'load_freq_dawg', b'0')):
            lib.TessBaseAPISetVariable(self.ocr, var, val)
        self.ocr_lang = ocr_lang
        self.tessdata_path = tessdata
        _log(f"  OCR: {ocr_lang}  tessdata={os.path.basename(tessdata)}  rc={rc}"
             f"  ({(time.perf_counter()-t0):.1f}s)")
        self.ready = True
        from performance_runtime import prepare_runtime
        prepare_runtime(self)
        return self

    def set_lang(self, lang='chi_sim'):
        """运行时切换 OCR 语言(新建一个引擎句柄再替换, 避免原地重init卡住)。

        ⚠️ 重要实测结论: 用 'chi_sim+eng' 组合语言时, 中文会被强行识别成拉丁乱码
        (如「中华人民共和国」→ 'Wie AD dk Bete')。中文场景必须用**纯 chi_sim**
        —— 它本身也包含 ASCII 字符, 英文同样能认。"""
        lang = (lang or 'chi_sim').strip()
        from performance_runtime import close_runtime, prepare_runtime
        close_runtime(self)
        try:
            lib = self.ocr_lib
            td = getattr(self, 'tessdata_path', None) or TESSDATA_STD
            newh = lib.TessBaseAPICreate()
            if not newh:
                _log("切换语言失败: 创建句柄失败")
                return False
            rc = lib.TessBaseAPIInit3(newh, td.encode(), lang.encode())
            if rc != 0:
                try:
                    lib.TessBaseAPIDelete(newh)
                except Exception:
                    pass
                _log(f"切换语言失败 rc={rc}")
                return False
            lib.TessBaseAPISetPageSegMode(newh, 6)
            for var, val in ((b'load_system_dawg', b'0'), (b'load_freq_dawg', b'0')):
                lib.TessBaseAPISetVariable(newh, var, val)
            old = getattr(self, 'ocr', None)
            self.ocr = newh
            self.ocr_lang = lang
            self._smart_cache = None
            prepare_runtime(self)
            try:
                if old:
                    lib.TessBaseAPIDelete(old)
            except Exception:
                pass
            _log(f"OCR 语言 → {lang}" + ("  ⚠️ 组合语言可能让中文变乱码" if '+' in lang else ""))
            return True
        except Exception as e:
            _log(f"切换语言异常: {type(e).__name__}: {e}")
            return False

    def status(self):
        _log(f"模块版本: {VERSION}")
        if not self.ready:
            _log("状态: 未初始化 (第一次调用 run_* 会自动初始化)")
            return
        _log(f"状态: 已就绪")
        _log(f"  OCR 语言 : {self.ocr_lang}")
        _log(f"  PL 分辨率: {self.plw}x{self.plh}")
        _log(f"  已处理   : {self.images} 张/帧, 累计 {self.total_ms/1000:.1f} 秒"
             f" (平均 {self.total_ms/max(self.images,1):.0f} ms)")

    # ---------------- PL 预处理 ----------------
    def _reset_dma(self):
        self.dmmio.write(MM2S_DMACR, 4)
        self.dmmio.write(S2MM_DMACR, 4)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            if (self.dmmio.read(MM2S_DMASR) & 0x1) and (self.dmmio.read(S2MM_DMASR) & 0x1):
                break
            time.sleep(0.001)

    def preprocess(self, gray):
        """灰度图 -> PL(二值化+形态学) -> 二值图. 返回 (binary, 耗时明细dict)"""
        h, w = gray.shape
        if w > MAXW or h > MAXH:
            s = min(MAXW / w, MAXH / h)
            gray = cv2.resize(gray, (max(2, int(w * s)) // 2 * 2, max(2, int(h * s)) // 2 * 2),
                              interpolation=cv2.INTER_AREA)
            h, w = gray.shape
        self.plw, self.plh = w, h
        N, NB = w * h, w * h * 4
        t0 = time.perf_counter()
        self.inbuf[:N] = self.LUT[gray].reshape(-1)
        t_pack = (time.perf_counter() - t0) * 1000
        self._reset_dma()
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.bin_mmio.write(ARG0, int(thr))
        self.bin_mmio.write(REG_W, w); self.bin_mmio.write(REG_H, h); self.bin_mmio.write(REG_EN, 1)
        self.mor_mmio.write(ARG0, self.morph_mode)
        self.mor_mmio.write(REG_W, w); self.mor_mmio.write(REG_H, h); self.mor_mmio.write(REG_EN, 1)
        self.dmmio.write(S2MM_DMACR, 1)
        self.dmmio.write(S2MM_DA, self.outbuf.device_address)
        self.dmmio.write(S2MM_LENGTH, NB)
        self.dmmio.write(MM2S_DMACR, 1)
        self.dmmio.write(MM2S_SA, self.inbuf.device_address)
        self.dmmio.write(MM2S_LENGTH, NB)
        t0 = time.perf_counter()
        self.bin_mmio.write(CTRL, 1); self.mor_mmio.write(CTRL, 1)
        while not ((self.dmmio.read(MM2S_DMASR) & 0x2) and (self.dmmio.read(S2MM_DMASR) & 0x2)):
            if time.perf_counter() - t0 > 15:
                return None, {'pl': None}
        t_pl = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        binary = ((self.outbuf[:N] >> 16) & 0xFF).astype(np.uint8).reshape(h, w)
        t_read = (time.perf_counter() - t0) * 1000
        return binary, {'pack': t_pack, 'pl': t_pl, 'read': t_read, 'thresh': int(thr)}

    # ---------------- 条码 / ROI / OCR ----------------
    def barcodes(self, gray, binary=None, accurate=True):
        """多方案解码条码(提高检出率):
           原图 -> 放大2x/3x -> CLAHE -> 锐化 -> PL二值图 -> Otsu -> 自适应阈值 -> 闭运算 -> 反相
           每个方案得到的坐标都会换算回原图尺寸; 结果去重并做校验和验证。
           返回 ([(类型,内容,rect)], 耗时ms, 生效方案名)"""
        from pyzbar import pyzbar
        t0 = time.perf_counter()
        h, w = gray.shape
        maxdim = max(h, w)
        small_img = maxdim < 1400          # 小图才值得放大(大图放大既慢又无用)
        attempts = []

        def add(name, im, scale=1.0):
            if im is not None:
                attempts.append((name, im, scale))

        # 便宜的先做; 大图不做放大
        add('原图', gray)
        if binary is not None:
            # ⭐ PL 二值图可能被缩放(>PL上限时), 坐标要换算回原图尺度, 否则框会偏
            _sc_bin = (float(binary.shape[1]) / gray.shape[1]) if binary.shape[1] != gray.shape[1] else 1.0
            add('PL二值图', binary, _sc_bin)
        try:
            bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            add('Otsu', bw)
        except Exception:
            bw = None
        if small_img:
            add('放大2x', cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC), 2.0)
        if accurate:
            # 大图: 其余方案改在缩小图上做(否则每种方案都要几百毫秒, 总计好几秒)
            if maxdim > 1400:
                s = 1400.0 / maxdim
                work = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                wsc = s
                wbin = (cv2.resize(binary, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                        if binary is not None else None)
            else:
                work, wsc, wbin = gray, 1.0, binary
            try:
                add('自适应阈值', cv2.adaptiveThreshold(work, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                       cv2.THRESH_BINARY, 31, 10), wsc)
            except Exception:
                pass
            if small_img:
                add('放大3x', cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC), 3.0)
            try:
                add('CLAHE', clahe(work), wsc)
                add('锐化', unsharp(work), wsc)
                if wbin is not None:
                    add('闭运算', cv2.morphologyEx(wbin, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)), wsc)
                add('反相', cv2.bitwise_not(work), wsc)
            except Exception:
                pass

        found = {}          # (类型,内容) -> {'count','rect','src'}
        tried = 0
        for name, im, scale in attempts:
            try:
                bars = pyzbar.decode(im, symbols=SYMS) if SYMS else pyzbar.decode(im)
            except Exception:
                bars = []
            tried += 1
            for c in bars:
                try:
                    data = c.data.decode('utf-8', 'ignore')
                except Exception:
                    continue
                if not data:
                    continue
                r = c.rect
                rect = (int(r.left / scale), int(r.top / scale),
                        int(r.width / scale), int(r.height / scale))
                e = found.setdefault((c.type, data), {'count': 0, 'rects': [], 'src': name})
                e['count'] += 1
                e['rects'].append(rect)      # ⭐ 收集所有方案的框, 最后取并集(框更完整)
            # 提前结束: 已经解出校验通过的条码, 且试过至少 3 种方案
            if found and tried >= 3 and any(self.bar_checksum(t, d) is True for (t, d) in found):
                break
            if found and not accurate:
                break       # 快速模式: 一有结果就停

        # ⭐ 补框重解: 校验和失败的条码, 说明可能只框住了一部分;
        #    用"外扩 + 放大"的局部图再解一次, 成功则用更完整的框替换
        for (typ, data) in list(found.keys()):
            if self.bar_checksum(typ, data) is not False:
                continue
            x0, y0, rw, rh = union_rect(found[(typ, data)]['rects'])
            px, py = int(rw * 0.30) + 8, int(rh * 0.80) + 8
            x0 = max(0, x0 - px)
            y0 = max(0, y0 - py)
            x1 = min(gray.shape[1], x0 + rw + 2 * px)
            y1 = min(gray.shape[0], y0 + rh + 2 * py)
            roi = gray[y0:y1, x0:x1]
            if roi.size < 100:
                continue
            for fsc in (2.0, 3.0):
                try:
                    up = cv2.resize(roi, None, fx=fsc, fy=fsc, interpolation=cv2.INTER_CUBIC)
                    bars2 = pyzbar.decode(up, symbols=SYMS) if SYMS else pyzbar.decode(up)
                except Exception:
                    bars2 = []
                got = False
                for c in bars2:
                    try:
                        d2 = c.data.decode('utf-8', 'ignore')
                    except Exception:
                        continue
                    if not d2 or self.bar_checksum(c.type, d2) is not True:
                        continue
                    r2 = c.rect
                    rect2 = (x0 + int(r2.left / fsc), y0 + int(r2.top / fsc),
                             int(r2.width / fsc), int(r2.height / fsc))
                    old = found.get((c.type, d2))
                    found[(c.type, d2)] = {
                        'count': (old['count'] if old else 0) + 1,
                        'rects': [rect2], 'src': '补框重解'}
                    got = True
                if got:
                    break

        # 排序: 校验通过的优先, 其次出现次数多、面积大的
        def _maxarea(e):
            return max((r[2] * r[3] for r in e['rects']), default=0)

        def rank(kv):
            (typ, data), e = kv
            chk = self.bar_checksum(typ, data)
            order = 0 if chk is True else (1 if chk is None else 2)
            return (order, -e['count'], -_maxarea(e))

        def union_rect(rects):
            """同一内容的多个检测框取并集 —— 解决"框不住完整条码"的问题。
               只并入与最大框有明显重叠的框, 避免个别异常框把范围撑大。"""
            if not rects:
                return (0, 0, 0, 0)
            big = max(rects, key=lambda r: r[2] * r[3])
            x0, y0 = big[0], big[1]
            x1, y1 = big[0] + big[2], big[1] + big[3]
            ba = max(big[2] * big[3], 1)
            for r in rects:
                if r is big:
                    continue
                rx0, ry0 = r[0], r[1]
                rx1, ry1 = r[0] + r[2], r[1] + r[3]
                ix = max(0, min(x1, rx1) - max(x0, rx0))
                iy = max(0, min(y1, ry1) - max(y0, ry0))
                if ix * iy > 0.35 * min(r[2] * r[3], ba):
                    x0, y0 = min(x0, rx0), min(y0, ry0)
                    x1, y1 = max(x1, rx1), max(y1, ry1)
            return (x0, y0, x1 - x0, y1 - y0)

        self._bar_info = {}
        out = []
        for (typ, data), e in sorted(found.items(), key=rank):
            chk = self.bar_checksum(typ, data)
            rect = union_rect(e['rects'])
            # 对称轻微外扩(让绿框完整覆盖条码, 同时不把框撑得比例失真), 并限制在画面内
            px = max(2, int(rect[2] * 0.05))
            py = max(2, int(rect[3] * 0.05))
            bx0 = max(0, rect[0] - px)
            by0 = max(0, rect[1] - py)
            bx1 = min(gray.shape[1], rect[0] + rect[2] + px)
            by1 = min(gray.shape[0], rect[1] + rect[3] + py)
            rect = (bx0, by0, max(1, bx1 - bx0), max(1, by1 - by0))
            self._bar_info[(typ, data)] = {'count': e['count'], 'check': chk, 'src': e['src']}
            out.append((typ, data, rect))
        srcs = sorted(set(e['src'] for e in found.values()))
        return out, (time.perf_counter() - t0) * 1000, (', '.join(srcs) if srcs else '未检出')

    @staticmethod
    def bar_checksum(typ, data):
        """EAN13/EAN8/UPC 校验和验证; 返回 True/False/None(不适用)"""
        try:
            if typ in ('EAN13', 'UPCA') and len(data) >= 13 and data[:13].isdigit():
                d = [int(c) for c in data[:13]]
                s = sum(d[i] * (1 if i % 2 == 0 else 3) for i in range(12))
                return (10 - s % 10) % 10 == d[12]
            if typ == 'EAN8' and len(data) >= 8 and data[:8].isdigit():
                d = [int(c) for c in data[:8]]
                s = sum(d[i] * (3 if i % 2 == 0 else 1) for i in range(7))
                return (10 - s % 10) % 10 == d[7]
        except Exception:
            pass
        return None

    def print_summary(self, R, title=''):
        """打印结果汇总块(条码 / 文字 / 耗时)"""
        bc = R.get('bar_counts') or {}
        tc = R.get('text_counts') or {}
        nf = R.get('nframes') or 0
        _log("=" * 68)
        _log(f"  📊 识别汇总   {title}" if title else "  📊 识别汇总")
        _log("=" * 68)
        _log(f"  ▌条码  ({len(R.get('bars', []))} 项)   解码方式: {R.get('bar_src', '-')}")
        if R.get('bars'):
            for i, (t, d, _r) in enumerate(R['bars'], 1):
                info = getattr(self, '_bar_info', {}).get((t, d), {})
                chk = info.get('check', self.bar_checksum(t, d))
                mark = '✅ 校验通过' if chk is True else ('⚠️ 校验失败' if chk is False else '')
                if t == 'EAN13' and d[:3] in ('978', '979'):
                    mark += f'   (ISBN-13: {d[:3]}-{d[3:6]}-{d[6:9]}-{d[9:12]}-{d[12:]})'
                cnt = bc.get((t, d))
                occ = f"   出现 {cnt}/{nf} 次" if (cnt and nf) else ''
                _log(f"    [{i}] {t:<8} {d}   {mark}{occ}")
        else:
            _log("    (未检出条码)")
        _log(f"  ▌文字  ({len(R.get('texts', []))} 行)")
        if R.get('texts'):
            for i, t in enumerate(R['texts'], 1):
                cnt = tc.get(t)
                occ = f"   [出现 {cnt}/{nf} 次]" if (cnt and nf) else ''
                _log(f"    [{i}] {t}{occ}")
        elif R.get('cached'):
            _log("    (画面未变化, 沿用上一帧结果)")
        else:
            _log("    (未识别到文字)")
        cf = R.get('conf', -1)
        if cf is not None and cf >= 0:
            vw = R.get('views', '')
            _log(f"  ▌结果质量分  {cf}  (有效字符-符号惩罚+行数奖励, 越大越好)"
                 + (f"   视图: {vw}" if vw else ''))
        # 条码为主画面时, 文字多为误识别 -> 明确提示
        br = R.get('bar_ratio', 0) or 0
        if br > 0.15 and R.get('texts'):
            _log(f"  ⚠️ 提示: 条码占画面 {br:.0%}, 上面的文字很可能是误识别"
                 f"(此类图建议用 do_ocr=False 只做条码)")
        if R.get('total') is not None:
            _log(f"  ▌耗时  PL {R.get('pl', 0):.2f}ms | 条码 {R.get('t_bar', 0):.0f}ms | "
                 f"ROI {R.get('t_roi', 0):.0f}ms | OCR {R.get('t_ocr', 0):.0f}ms"
                 f"  → 合计 {R.get('total', 0):.0f}ms")
        # 一行速览: 让结果一眼可见(条码内容 / 文字行数)
        if R.get('bars'):
            bsum = ' | '.join(f"{t} {d}" for t, d, _r in R['bars'][:2])
        else:
            bsum = '未检测到条码'
        _log(f"  ▶ 速览:  条码 → {bsum}   |   文字 {len(R.get('texts', []))} 行"
             + (f"   (已滤除 {self._nongraphic} 个图形区)" if getattr(self, '_nongraphic', 0) else ''))
        _log("=" * 68)

    def is_text_like(self, binary, x, y, w, h, strict=False):
        """判断一块区域像不像"文字行": 文字行由多个高度相近的小连通域排成一行;
           插图/电路图则是少量大而不规则的块 -> 用来滤掉图形区的误识别

           strict=True (摄像头实景推荐): 额外用"笔画粗细/行投影起伏/实心块"判据,
           可以滤掉背景里的暗色物体、阴影、色块等非文字区域"""
        roi = binary[max(0, y):y + h, max(0, x):x + w]
        if roi.size < 100:
            return False
        hh, ww = roi.shape[:2]
        if strict:
            # ① 长宽比: 文字行偏扁长
            if ww / max(hh, 1) < 1.5:
                return False
            fg = (roi == 0)                       # 前景 = 暗像素
            # ② 密度: 实心物体密度接近 1; 噪声密度接近 0
            fill = float(fg.mean())
            if fill > 0.42 or fill < 0.02:
                return False
            # ③ 笔画粗细: 水平前景游程的平均长度应远小于行高
            n_run = int(np.count_nonzero(fg[:, 0])) + \
                int(np.count_nonzero(fg[:, 1:] & ~fg[:, :-1]))
            if n_run <= 0:
                return False
            mean_run = float(fg.sum()) / n_run
            if mean_run > 0.38 * max(hh, 1):
                return False
            # ④ 行投影要有起伏(文字行有峰谷; 实心块是平的)
            prof = fg.sum(axis=1).astype(np.float32)
            if prof.mean() < 1e-6:
                return False
            if float(prof.std() / (prof.mean() + 1e-6)) < 0.22:
                return False
        dark = cv2.bitwise_not(roi)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        if n <= 2:
            return False
        areas = stats[1:, 4]
        hs = stats[1:, 3][areas >= 4]        # 只统计有意义的连通域高度
        if len(hs) < 3:
            return False
        med = float(np.median(hs))
        if med < 4:
            return False
        # 高度接近中位数的比例(中文大字尺寸有差异, 阈值放宽)
        close = float(np.mean(np.abs(hs - med) <= 0.9 * med))
        # 连通域宽度也不能太大(插图里常有大块)
        ws = stats[1:, 2][areas >= 4]
        wide = float(np.mean(ws > 0.7 * w)) if len(ws) else 0.0
        if not (close >= 0.3 and wide <= 0.5):
            return False
        if strict:
            # ⑤ 单个连通域不能占太大(背景大块物体)
            if float(areas.max()) > 0.30 * roi.size:
                return False
            # ⑥ 连通域太少且有大块 -> 更像物体而不是文字
            if len(hs) < 4 and float(areas.max()) > 0.15 * roi.size:
                return False
        return True

    def roi_text_score(self, binary, rect, gray=None):
        """给候选区域打"像文字"的分 —— 用于挑选 ROI(而不是按面积挑)。
           面积大的往往是暗色物体/暗角/阴影, 按面积挑就会一直框错地方。
           返回分值, 越大越像文字。"""
        x, y, w, h = rect
        if w < 8 or h < 8:
            return -9.0
        roi = binary[y:y + h, x:x + w]
        if roi.size < 100:
            return -9.0
        fg = (roi == 0)                      # 前景 = 暗像素
        fill = float(fg.mean())
        score = 0.0
        # ① 密度: 文字行通常 3%~40%
        if 0.03 <= fill <= 0.40:
            score += 1.0
        elif fill > 0.55:
            score -= 2.0                     # 实心块
        # ② 笔画粗细: 文字笔画细
        n_run = int(np.count_nonzero(fg[:, 0])) + \
            int(np.count_nonzero(fg[:, 1:] & ~fg[:, :-1]))
        if n_run > 0:
            ratio = (float(fg.sum()) / n_run) / max(h, 1)
            if ratio <= 0.30:
                score += 1.2
            elif ratio > 0.45:
                score -= 1.5                 # 粗大块
        # ③ 行投影起伏: 文字行有峰谷
        prof = fg.sum(axis=1).astype(np.float32)
        if prof.mean() > 1e-6:
            score += min(float(prof.std() / (prof.mean() + 1e-6)), 0.8) * 1.5
        # ④ 连通域数量 + 高度一致性(文字由多个相近大小的字组成)
        #    注意: 数量奖励要"封顶且温和", 否则会偏向密排小字, 把大标题挤掉
        try:
            dark = cv2.bitwise_not(roi)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
            areas = stats[1:, 4]
            hs = stats[1:, 3][areas >= 4]
            nc = len(hs)
            if nc >= 3:
                score += min(nc, 8) / 8.0 * 0.7        # 最多 +0.7
                med_h = float(np.median(hs))
                score += float(np.mean(np.abs(hs - med_h) <= 0.9 * med_h)) * 0.8
                # ⭐ 大字号奖励: 标题类大字(字高占画面高度 6% 以上)应优先
                if gray is not None:
                    fh = gray.shape[0]
                    if med_h >= 0.06 * fh:
                        score += 1.6
                    elif med_h >= 0.04 * fh:
                        score += 0.8
            if nc and float(areas.max()) > 0.30 * roi.size:
                score -= 1.5                 # 单个大块
        except Exception:
            pass
        # ⑤ 长宽比: 文字行偏扁长
        ar = w / max(h, 1)
        if 2.0 <= ar <= 25:
            score += 0.8
        elif ar < 1.4:
            score -= 1.0
        # ⑥ 背景亮度: 暗色物体/暗角 ROI 整体明显偏暗 -> 重罚
        if gray is not None:
            try:
                gm = float(gray[y:y + h, x:x + w].mean())
                fm = float(gray.mean()) + 1e-6
                r = gm / fm
                if r < 0.55:
                    score -= 2.5
                elif r >= 0.80:
                    score += 0.8
            except Exception:
                pass
        # ⑦ 贴边/角落: 摄像头画面的四角边缘多是背景噪声(晾衣架、阴影、亮斑),
        #    实测角上的误检最多, 所以这里给较重的惩罚
        if gray is not None:
            try:
                H, W = gray.shape[:2]
                m = min(x, y, W - (x + w), H - (y + h))
                if m < 0.02 * min(H, W):
                    score -= 2.6                     # 紧贴画面边缘
                elif m < 0.05 * min(H, W):
                    score -= 1.6
                # 四角区域额外再罚(角上最脏)
                cx = x + w / 2.0
                cy = y + h / 2.0
                in_corner = ((cx < 0.28 * W or cx > 0.72 * W) and
                             (cy < 0.30 * H or cy > 0.70 * H))
                if in_corner:
                    score -= 1.8
            except Exception:
                pass
        return score

    def local_binary_full(self, gray, max_w=960):
        """整图局部(自适应)二值化 —— 作为文字检测的输入。

        为什么需要它: PL 端是"固定全局阈值"的硬件二值化, 对光照不均/整体偏亮偏暗的
        区域(证件、卡片、屏幕、阴影覆盖处)会整块判成背景, 导致该区域一个字都测不出。
        局部二值化按像素邻域自适应, 在这类区域依然能分出笔画。
        (PL 的全局二值化仍用于条码检测与形态学流水线; 这里只是给文字检测补一路输入)
        """
        H, W = gray.shape[:2]
        sc = min(1.0, max_w / float(W))
        g = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else gray
        blk = max(15, (g.shape[1] // 28) | 1)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blk, 10)
        try:
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        except Exception:
            pass
        if sc < 1.0:
            bw = cv2.resize(bw, (W, H), interpolation=cv2.INTER_NEAREST)
        return bw

    def local_binarize(self, g):
        """对单个文字块做局部二值化 —— 全局阈值在局部低对比度区域(证件/卡片/反光)
           可能整块变成白色, 导致该区域一个字都识别不出。低对比度时改用自适应阈值。"""
        if g is None or g.size < 100:
            return g
        try:
            rng = int(g.max()) - int(g.min())
            if rng < 45:
                blk = max(15, (min(g.shape[:2]) // 2) | 1)
                bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY, blk, 8)
                fg = np.count_nonzero(bw == 0) / float(bw.size)
                if fg < 0.008 or fg > 0.70:
                    bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    fg2 = np.count_nonzero(bw2 == 0) / float(bw2.size)
                    if 0.008 <= fg2 <= 0.70:
                        bw = bw2
                return bw
            return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        except Exception:
            try:
                return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            except Exception:
                return g

    def tighten_roi(self, binary, rect, margin=3):
        """把 ROI 收紧到前景的实际范围 —— 让画出来的框更贴合文字(解决"框不准")。
           用行/列投影并设一个小阈值, 比直接取 min/max 更抗孤立噪点。"""
        x, y, w, h = rect
        if w < 4 or h < 4:
            return rect
        roi = binary[y:y + h, x:x + w]
        fg = (roi == 0)
        if not fg.any():
            return rect
        colsum = fg.sum(axis=0)
        rowsum = fg.sum(axis=1)
        cth = max(1, int(0.03 * int(colsum.max())))
        rth = max(1, int(0.03 * int(rowsum.max())))
        cols = np.nonzero(colsum >= cth)[0]
        rows = np.nonzero(rowsum >= rth)[0]
        if len(cols) == 0 or len(rows) == 0:
            return rect
        x0 = x + int(cols[0]) - margin
        x1 = x + int(cols[-1]) + 1 + margin
        y0 = y + int(rows[0]) - margin
        y1 = y + int(rows[-1]) + 1 + margin
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(binary.shape[1], x1)
        y1 = min(binary.shape[0], y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return rect
        return (x0, y0, x1 - x0, y1 - y0)

    def text_rois_mser(self, gray, max_roi=3, max_h_ratio=0.25, work_w=640):
        """用 MSER 在灰度图上检测字符候选, 再按行聚类成文字行 ROI。

        相比"全局二值图 + 膨胀", 这个方法的优势:
          · 基于局部对比度 → 对光照不均/暗角/阴影鲁棒(平滑区域找不到任何候选)
          · 得到的是字符级区域 → 聚类出来的框天然贴合文字, 不会框到边缘
        找不到候选时返回 [] (调用方回退到二值图方法)。
        """
        H, W = gray.shape[:2]
        # 缩到 work_w 宽度做检测(提速), 最后把框换算回原尺度
        sc = min(1.0, work_w / float(W))
        small = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1.0 else gray
        h2, w2 = small.shape[:2]
        try:
            mser = cv2.MSER_create()
            try:
                mser.setDelta(5)
                mser.setMinArea(max(8, int(0.00002 * w2 * h2)))
                mser.setMaxArea(int(0.02 * w2 * h2))
                mser.setMaxVariation(0.55)
            except Exception as e:
                self._mser_dbg = f'set-{type(e).__name__}'
            _regions, boxes = mser.detectRegions(small)
            boxes = np.asarray(boxes)
        except Exception as e:
            self._mser_dbg = f'detect-{type(e).__name__}:{e}'
            return []
        if boxes is None or len(boxes) < 4:
            self._mser_dbg = f'few({0 if boxes is None else len(boxes)})'
            return []
        self._mser_dbg = f'raw{len(boxes)}'
        # ---- 候选筛选: 尺寸/长宽比/局部对比度 ----
        cands = []
        for (x, y, bw, bh) in boxes:
            if bw < 3 or bh < 6 or bh > max_h_ratio * h2:
                continue
            ar = bw / float(bh)
            if ar > 7 or ar < 0.06:
                continue
            pad = max(2, bh // 2)
            y0 = max(0, y - pad); y1 = min(h2, y + bh + pad)
            x0 = max(0, x - pad); x1 = min(w2, x + bw + pad)
            inner = float(small[y:y + bh, x:x + bw].mean())
            around = float(small[y0:y1, x0:x1].mean())
            if inner > 0.90 * around:          # 不够暗 = 不是深色文字
                continue
            cands.append((int(x), int(y), int(bw), int(bh)))
        if len(cands) < 4:
            self._mser_dbg += f'/cand{len(cands)}'
            return []
        # ---- 按"高度相近 + 垂直中心接近 + 水平距离不远"聚成行 ----
        cands.sort(key=lambda c: c[1])
        used = [False] * len(cands)
        lines = []
        for i in range(len(cands)):
            if used[i]:
                continue
            grp = [cands[i]]
            used[i] = True
            hi = cands[i][3]
            changed = True
            while changed:                     # 反复吸收, 直到不再变大
                changed = False
                gx0 = min(g[0] for g in grp); gy0 = min(g[1] for g in grp)
                gx1 = max(g[0] + g[2] for g in grp); gy1 = max(g[1] + g[3] for g in grp)
                gcy = (gy0 + gy1) / 2.0
                for j in range(len(cands)):
                    if used[j]:
                        continue
                    d = cands[j]
                    if not (0.5 <= d[3] / max(hi, 1) <= 2.0):
                        continue
                    if abs((d[1] + d[3] / 2.0) - gcy) > 0.7 * max(hi, d[3]):
                        continue
                    gap = max(0, max(gx0, d[0]) - min(gx1, d[0] + d[2]))
                    if gap > 2.5 * max(hi, d[3]):
                        continue
                    grp.append(d)
                    used[j] = True
                    changed = True
            gx0 = min(g[0] for g in grp); gy0 = min(g[1] for g in grp)
            gx1 = max(g[0] + g[2] for g in grp); gy1 = max(g[1] + g[3] for g in grp)
            gw, gh = gx1 - gx0, gy1 - gy0
            # 真文字行: 至少 3 个字符、宽度和高度够
            if len(grp) >= 3 and gw >= 0.04 * w2 and gh >= 8:
                lines.append((gx0, gy0, gw, gh, len(grp)))
        if not lines:
            return []
        # 字符多的更像文字行; 换算回原图尺度。顺带丢掉"被已选框大部分包含"的重复检测
        lines.sort(key=lambda t: -t[4])
        picked = []
        for (x, y, w, h, n) in lines:
            dup = False
            for (px, py, pw, ph, _pn) in picked:
                ix = max(0, min(x + w, px + pw) - max(x, px))
                iy = max(0, min(y + h, py + ph) - max(y, py))
                if ix * iy > 0.55 * (w * h):
                    dup = True
                    break
            if dup:
                continue
            picked.append((x, y, w, h, n))
            if len(picked) >= max_roi:
                break
        out = []
        for (x, y, w, h, _n) in picked:
            out.append((int(x / sc), int(y / sc), int(w / sc), int(h / sc)))
        return out

    def text_rois(self, binary, excl=(), max_roi=3, max_h_ratio=0.25, accurate=False,
                  strict_text=False, gray=None):
        """找出文字区域。accurate=True 时放宽阈值(减少漏检), 代价是 ROI 变多、耗时增加
           会用 is_text_like 过滤掉插图/图形区(避免 tesseract 在图形上瞎猜)
           strict_text=True: 所有模式下都做文字形状校验(摄像头实景推荐, 滤掉背景物体)"""
        PLW, PLH = self.plw, self.plh
        if accurate:
            max_h_ratio = max(max_h_ratio, 0.35)
        dark = cv2.bitwise_not(binary)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
        areas = stats[:, 4]
        min_area = 10 if accurate else 15
        max_area = (0.06 if accurate else 0.04) * PLW * PLH
        ok = (areas >= min_area) & (areas < max_area) & \
             (stats[:, 2] < 0.92 * PLW) & (stats[:, 3] < 0.6 * PLH)
        ok[0] = False
        keep = np.where(ok[lab], np.uint8(255), np.uint8(0))
        k = max(18 if accurate else 20, PLW // (34 if accurate else 28))
        lines = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
        cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        min_w = min(40 if accurate else 60, PLW * (0.04 if accurate else 0.06))
        min_h = min(10 if accurate else 12, PLH * (0.018 if accurate else 0.025))
        dmin, dmax = (0.015, 0.60) if accurate else (0.025, 0.55)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < min_w or h < min_h or h > PLH * max_h_ratio or w / max(h, 1) < 1.1:
                continue
            roi = binary[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            dens = float(np.count_nonzero(roi == 0)) / roi.size
            if dens < dmin or dens > dmax:
                continue
            # 只在"与条码区重叠较多"时才排除(避免把紧贴条码上方的 ISBN 行也误排除)
            skip = False
            for e in excl:
                ix = max(0, min(x + w, e[0] + e[2]) - max(x, e[0]))
                iy = max(0, min(y + h, e[1] + e[3]) - max(y, e[1]))
                if ix * iy > 0.45 * w * h:
                    skip = True
                    break
            if skip:
                continue
            if (accurate or strict_text) and \
                    not self.is_text_like(binary, x, y, w, h, strict=strict_text):
                self._nongraphic += 1
                continue
            out.append((x, y, w, h))
        # ⭐ 按"像文字的程度"排序挑选(不再按面积大小!) —— 面积最大的常是暗色物体/暗角
        scored = [(self.roi_text_score(binary, r, gray), r) for r in out]
        scored.sort(key=lambda t: -t[0])
        # 去掉被"更优的更大 ROI"完全包含的重复框, 并丢掉明显不像文字的
        dedup = []
        for sc, r in scored:
            if sc < -0.5:
                continue
            x, y, w, h = r
            if any(x >= a[0] - 4 and y >= a[1] - 4 and x + w <= a[0] + a[2] + 4 and y + h <= a[1] + a[3] + 4
                   for a in dedup):
                continue
            dedup.append(r)
            if len(dedup) >= max_roi:
                break
        # ⭐ 收紧到文字实际范围, 让框更贴合
        return [self.tighten_roi(binary, r) for r in dedup]

    def make_canvas(self, binary, rois, cropmax=450):
        crops = []
        for (x, y, w, h) in rois:
            c = binary[max(0, y - 8):min(self.plh, y + h + 8), max(0, x - 8):min(self.plw, x + w + 8)]
            if c.size < 200:
                continue
            if c.shape[1] > cropmax:
                s = cropmax / c.shape[1]
                c = cv2.resize(c, (cropmax, max(8, int(c.shape[0] * s))))
            crops.append(c)
        if not crops:
            return None
        gap, wmax = 30, max(c.shape[1] for c in crops)
        canvas = np.full((sum(c.shape[0] for c in crops) + gap * (len(crops) - 1), wmax), 250, np.uint8)
        yy = 0
        for c in crops:
            canvas[yy:yy + c.shape[0], 0:c.shape[1]] = c
            yy += c.shape[0] + gap
        return canvas

    def ocr_img(self, img):
        h, w = img.shape
        t0 = time.perf_counter()
        self.ocr_lib.TessBaseAPISetImage(self.ocr, img.tobytes(), w, h, 1, w)
        ptr = self.ocr_lib.TessBaseAPIGetUTF8Text(self.ocr)
        t = ctypes.string_at(ptr).decode('utf-8', 'ignore')
        self.ocr_lib.TessDeleteText(ptr)
        return [l.strip() for l in t.split('\n') if len(l.strip()) > 1], (time.perf_counter() - t0) * 1000

    def ocr_one(self, img, psm=7, dawg=True, whitelist=None):
        """识别单块区域, 返回 (文本行列表, 平均置信度0~100, 耗时ms)"""
        try:
            if dawg:
                self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'load_system_dawg', b'1')
                self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'load_freq_dawg', b'1')
            else:
                self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'load_system_dawg', b'0')
                self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'load_freq_dawg', b'0')
            self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'tessedit_char_whitelist', b'')
            if whitelist:
                self.ocr_lib.TessBaseAPISetVariable(self.ocr, b'tessedit_char_whitelist',
                                                    whitelist.encode())
            self.ocr_lib.TessBaseAPISetPageSegMode(self.ocr, psm)
            h, w = img.shape
            t0 = time.perf_counter()
            self.ocr_lib.TessBaseAPISetImage(self.ocr, img.tobytes(), w, h, 1, w)
            ptr = self.ocr_lib.TessBaseAPIGetUTF8Text(self.ocr)
            txt = ctypes.string_at(ptr).decode('utf-8', 'ignore')
            self.ocr_lib.TessDeleteText(ptr)
            conf = -1.0
            try:
                conf = round(float(self.ocr_lib.TessBaseAPIMeanTextConf(self.ocr)), 1)
            except Exception:
                pass
            return [l.strip() for l in txt.split('\n') if l.strip()], conf, \
                   (time.perf_counter() - t0) * 1000
        except Exception as e:
            return [], -1, 0.0
        finally:
            try:
                self.ocr_lib.TessBaseAPISetPageSegMode(self.ocr, 6)
            except Exception:
                pass

    def ocr_best(self, img, min_conf=0, whitelist=None):
        """同一块区域分别用 PSM6(多行) 与 PSM7(单行) 识别, 取"质量分"更高的结果。
           注: 不用 tesseract 的置信度 API(实测返回不可靠), 用自己算的 text_score"""
        lines6, _c6, t6 = self.ocr_one(img, psm=6, dawg=True, whitelist=whitelist)
        keep6 = [l for l in lines6 if not is_garbage(l)]
        s6 = text_score(keep6)
        # 块不高(单行)且已识别出内容 -> 不必再试单行模式
        if keep6 and img.shape[0] <= 90:
            return keep6, s6, t6
        lines7, _c7, t7 = self.ocr_one(img, psm=7, dawg=True, whitelist=whitelist)
        keep7 = [l for l in lines7 if not is_garbage(l)]
        s7 = text_score(keep7)
        if s7 > s6:
            return keep7, s7, t6 + t7
        return keep6, s6, t6 + t7

    # ---------------- 方向识别 ----------------
    def detect_rotation(self, frame):
        """用 tesseract OSD 判断画面需要旋转多少度(约 15 秒, 每次运行只做一次)
           返回 cv2 旋转码 或 None"""
        import re
        try:
            import pytesseract
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            s = min(800, g.shape[1])
            gs = cv2.resize(g, (s, max(8, int(g.shape[0] * s / g.shape[1]))))
            _log("  [方向] tesseract OSD 判定中 ...")
            t0 = time.perf_counter()
            txt = pytesseract.image_to_osd(gs)
            m = re.search(r'Rotate:\s*(\d+)', txt)
            if m:
                ang = int(m.group(1))
                code = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(ang)
                _log(f"  [方向] OSD: 需旋转 {ang}度 -> {ROTNAME[code]}  ({(time.perf_counter()-t0):.0f}s)")
                return code
        except Exception as e:
            _log(f"  [方向] OSD 不可用({type(e).__name__}), 按不旋转处理")
        return None

    def _apply_rot(self, img, rot):
        return img if rot is None else cv2.rotate(img, rot)

    # ---------------- 单帧处理 ----------------
    def process(self, img_bgr, do_ocr=True, prev_thumb=None, gate_th=10.0,
                max_roi=3, cropmax=450, accurate=False, min_conf=0, drop_latin=False,
                strict_text=False, per_roi=False):
        kwargs = dict(do_ocr=do_ocr, prev_thumb=prev_thumb, gate_th=gate_th,
                      max_roi=max_roi, cropmax=cropmax, accurate=accurate,
                      min_conf=min_conf, drop_latin=drop_latin,
                      strict_text=strict_text, per_roi=per_roi)
        from performance_runtime import profile_result
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        if (not do_ocr or getattr(self, 'ocr_layout', 'auto') == 'legacy' or
                getattr(self, 'ocr_hook', None) is not None):
            result, thumb = self._process_legacy(img_bgr, **kwargs)
        else:
            from ocr_pipeline import process
            result, thumb = process(self, img_bgr, **kwargs)
        return profile_result(result, wall_start, cpu_start,
                              int(getattr(self, 'target_ms', 1000))), thumb

    def _process_legacy(self, img_bgr, do_ocr=True, prev_thumb=None, gate_th=10.0,
                max_roi=3, cropmax=450, accurate=False, min_conf=0, drop_latin=False,
                strict_text=False, per_roi=False):
        """处理一帧/一张图, 返回 (结果dict, 缩略图)
           accurate=True 启用高精度模式: 逐行识别(PSM7/6取优) + 预处理增强 + 置信度过滤
           strict_text=True 对所有候选 ROI 做文字形状校验(摄像头实景推荐)
           per_roi=True 每个文字框单独识别, 结果与框一一对应(返回 R['roi_texts'])"""
        self._nongraphic = 0
        R = {'bars': [], 'texts': [], 'gate': False, 'bar_src': '-'}
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

        binary, t = self.preprocess(gray)
        if binary is None:
            R['error'] = '硬件事务超时'
            return R, prev_thumb
        # ⭐ 统一坐标空间: PL 可能因超过 1920x1080 缩放过, 这里把二值图还原到
        #    与灰度图同样大小 —— 之后所有框(条码/文字/ROI)都在同一坐标系, 彻底避免错位
        if binary.shape[1] != gray.shape[1] or binary.shape[0] != gray.shape[0]:
            binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        R.update(t)

        # 条码解码放在 PL 之后: 可用 PL 二值图作为候选, 提升小条码/弱光照检出率
        R['bars'], R['t_bar'], R['bar_src'] = self.barcodes(gray, binary, accurate=accurate)
        # 条码占画面比例(用于提示"条码为主"的图, 文字结果可能是误识别)
        if R['bars']:
            _ba = sum(max(0, b[2][2]) * max(0, b[2][3]) for b in R['bars'])
            R['bar_ratio'] = _ba / float(max(gray.shape[0] * gray.shape[1], 1))

        # 变化门控
        thumb = cv2.resize(gray, (64, 48))
        changed = True
        if prev_thumb is not None:
            R['diff'] = float(cv2.absdiff(thumb, prev_thumb).mean())
            changed = R['diff'] > gate_th
        R['gate'] = not changed

        R['t_roi'] = R['t_ocr'] = 0.0
        R['nroi'] = 0
        R['cached'] = False
        R['conf'] = -1
        if do_ocr and changed:
            t0 = time.perf_counter()
            excl = [(b[2][0], b[2][1], b[2][2], b[2][3]) for b in R['bars']]
            # ⭐ 两套候选合并, 统一打分挑选:
            #    · 二值图法: 会沿水平方向膨胀, 能把一行字连成完整文字行(主力)
            #    · MSER 法: 基于局部对比度, 在光照不均时能补到漏检(辅助)
            #    只让某一套"有结果就用"是错的 —— MSER 常给出碎片, 那样必然识别乱码
            cands = []          # [(rect, src)]
            self._mser_dbg = ''
            # 文字检测用的二值图: 摄像头实景优先用局部二值化(PL 的固定全局阈值在
            # 偏亮/偏暗区域会整块失效), 失败则回退 PL 二值图
            tbin = binary
            if strict_text:
                try:
                    tbin = self.local_binary_full(gray)
                    R['roi_bin'] = 'local'
                except Exception:
                    tbin = binary
                    R['roi_bin'] = 'pl'
            else:
                R['roi_bin'] = 'pl'
            if gray is not None:
                try:
                    for r in self.text_rois_mser(gray, max_roi=4,
                                                 max_h_ratio=0.30 if accurate else 0.25):
                        cands.append((r, 'MSER'))
                except Exception as e:
                    self._mser_dbg = f'{type(e).__name__}'
            for r in self.text_rois(tbin, excl, max_roi=max(max_roi * 2, 4),
                                    accurate=accurate, strict_text=strict_text,
                                    gray=gray):
                cands.append((r, 'binary'))
            # 统一打分(都用同一张二值图 + 灰度亮度), 去掉与条码重叠太多的
            scored = []
            for (x, y, w, h), src in cands:
                skip = False
                for e in excl:
                    ix = max(0, min(x + w, e[0] + e[2]) - max(x, e[0]))
                    iy = max(0, min(y + h, e[1] + e[3]) - max(y, e[1]))
                    if ix * iy > 0.5 * w * h:
                        skip = True
                        break
                if skip:
                    continue
                scored.append((self.roi_text_score(tbin, (x, y, w, h), gray),
                               (x, y, w, h), src))
            scored.sort(key=lambda t: -t[0])
            rois = []
            for sc, r, src in scored:
                if sc < -0.5:
                    continue
                x, y, w, h = r
                # 去掉被已选框大部分包含的重复
                if any(max(0, min(x + w, a[0] + a[2]) - max(x, a[0])) *
                       max(0, min(y + h, a[1] + a[3]) - max(y, a[1])) > 0.55 * w * h
                       for a in rois):
                    continue
                rois.append(r)
                if len(rois) >= max_roi:
                    break
            R['roi_src'] = ('MSER+' if any(s == 'MSER' for _sc, _r, s in scored[:len(rois)])
                            else '') + 'binary'
            if self._mser_dbg:
                R['roi_src'] += f'({self._mser_dbg})'
            R['nroi'] = len(rois)
            # 二值图已还原到灰度尺寸, ROI 也在这个空间, 无需再换算
            R['rois'] = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rois]
            R['t_roi'] = (time.perf_counter() - t0) * 1000

            if per_roi and rois:
                # ⭐ 逐块识别 + 【结果质量反向过滤】:
                #    先按形状挑出较多候选, 逐个识别, 只保留"确实识别出有意义文字"的框。
                #    这样背景、图案、照片区因为识别不出文字会被自动淘汰 —— 从根上
                #    解决"总是框住背景/物体并识别出乱码"的问题。
                hook = getattr(self, 'ocr_hook', None)
                roi_texts = []
                good_rois = []
                n_ocr = 0
                for (rx, ry, rw, rh) in rois:
                    if len(good_rois) >= max_roi or n_ocr >= 6:
                        break
                    crop = binary[ry:ry + rh, rx:rx + rw]
                    if crop.size < 50:
                        continue
                    try:
                        f1 = np.count_nonzero(crop == 0) / float(crop.size)
                        if f1 < 0.004 or f1 > 0.80:
                            alt = self.local_binarize(gray[ry:ry + rh, rx:rx + rw])
                            if alt is not None and alt.size == crop.size:
                                f2 = np.count_nonzero(alt == 0) / float(alt.size)
                                if 0.004 <= f2 <= 0.80:
                                    crop = alt
                    except Exception:
                        pass
                    crop = np.ascontiguousarray(crop)
                    try:
                        prep, _n = prep_for_ocr(crop, target_h=36, border=16)
                    except Exception:
                        continue
                    n_ocr += 1
                    if hook is not None:
                        lines, dt = hook(prep)
                        R['t_ocr'] += dt
                    else:
                        lines, _sc, dt = self.ocr_best(prep)
                        R['t_ocr'] += dt
                    keep = [l for l in (lines or []) if l and l.strip() and not is_garbage(l)]
                    if not keep:
                        continue                      # 没识别出内容 -> 淘汰该框
                    total = ''.join(keep)
                    n_cjk = sum(1 for c in total if '\u4e00' <= c <= '\u9fff')
                    n_alnum = sum(1 for c in total if c.isalnum())
                    # 至少要有 1 个汉字, 或者 2 个以上字母数字(英文单词/编号)
                    if n_cjk < 1 and n_alnum < 2:
                        continue
                    if text_score(keep) < 1.0:
                        continue
                    good_rois.append((rx, ry, rw, rh))
                    roi_texts.append((n_cjk, keep[:3]))
                if good_rois:
                    # 汉字多的排前面(更像真实文字内容)
                    order = sorted(range(len(good_rois)),
                                   key=lambda i: (-roi_texts[i][0], good_rois[i][1]))
                    rois = [good_rois[i] for i in order]
                    roi_texts = [roi_texts[i][1] for i in order]
                else:
                    rois = []
                    roi_texts = []
                flat = []
                for ls in roi_texts:
                    flat.extend(ls)
                R['roi_texts'] = roi_texts
                R['rois'] = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rois]
                R['nroi'] = len(rois)          # 只保留"确实识别出内容"的框
                R['texts'] = merge_lines(flat, drop_latin=drop_latin)
                self._last_texts = list(R['texts'])
            elif accurate and rois:
                # ⭐ 高精度: 两个互补视图 + 质量分取优(比逐块识别又快又准)
                hook = getattr(self, 'ocr_hook', None)   # ← 供 v2 切换识别后端(默认 None)
                views = []
                canvas = self.make_canvas(binary, rois, cropmax=cropmax)
                if canvas is not None:
                    prep, _n = prep_for_ocr(canvas, target_h=36, border=16)
                    if hook is not None:
                        l1, dt1 = hook(prep)
                        s1 = text_score(l1)
                        R['t_ocr'] += dt1
                    else:
                        l1, s1, dt1 = self.ocr_best(prep)
                        R['t_ocr'] += dt1
                    views.append(('画布', l1, s1))
                if binary.shape[0] * binary.shape[1] <= 1_300_000:
                    w0 = binary.shape[1]
                    full = binary if w0 <= cropmax else cv2.resize(
                        binary, (cropmax, max(8, int(binary.shape[0] * cropmax / w0))))
                    prep2, _n2 = prep_for_ocr(full, target_h=36, border=16)
                    l2, s2, dt2 = self.ocr_best(prep2)
                    R['t_ocr'] += dt2
                    views.append(('整幅', l2, s2))
                # 逐块识别只在两个视图都没结果时才做(兜底, 较慢)
                if not any(text_score(v[1]) > 0 for v in views):
                    per = []
                    for (x, y, w, h) in rois:
                        crop = binary[max(0, y - 8):min(self.plh, y + h + 8),
                                      max(0, x - 8):min(self.plw, x + w + 8)]
                        if crop.size < 200:
                            continue
                        prep3, _n3 = prep_for_ocr(crop, target_h=36, border=16)
                        l3, s3, dt3 = self.ocr_best(prep3)
                        R['t_ocr'] += dt3
                        per.extend(l3)
                    views.append(('逐块', per, text_score(per)))
                views.sort(key=lambda v: -v[2])
                all_lines = []
                for _name, lines, _sc in views:      # 质量分高的视图在前
                    all_lines.extend(lines)
                merged = merge_lines(all_lines, drop_latin=drop_latin)
                R['texts'] = merged
                R['conf'] = round(max((v[2] for v in views), default=-1.0), 1)
                R['views'] = ', '.join(f"{n}:{s:.0f}" for n, _l, s in views)
                self._last_texts = list(merged)
            else:
                canvas = self.make_canvas(binary, rois, cropmax=cropmax)
                if canvas is not None:
                    hook = getattr(self, 'ocr_hook', None)   # ← 供 v2 切换识别后端(默认 None)
                    if hook is not None:
                        R['texts'], _t = hook(canvas)
                        R['t_ocr'] += _t
                    else:
                        R['texts'], R['t_ocr'] = self.ocr_img(canvas)
                    R['texts'] = merge_lines(R['texts'], drop_latin=drop_latin)
                    self._last_texts = list(R['texts'])
                else:
                    R['t_ocr'] += self._fallback_ocr(binary, cropmax, R, drop_latin=drop_latin)
        elif do_ocr and not changed:
            # ⭐ 门控跳过时复用上次文字结果(否则显示为空, 会让人以为识别失败)
            R['texts'] = list(getattr(self, '_last_texts', []))
            R['cached'] = True
        if R.get('conf', -1) in (-1, None) and R.get('texts'):
            R['conf'] = round(text_score(R['texts']), 1)
        R['total'] = (R.get('pack', 0) + R.get('pl', 0) + R.get('read', 0)
                      + R['t_bar'] + R['t_roi'] + R['t_ocr'])
        self.images += 1
        self.total_ms += R['total']
        return R, thumb

    def _process_tiled(self, img, **kw):
        """超大图分块处理: 按横向条带(+必要时纵向)切块, 每块在 PL 上限内,
           这样不必整体缩放 -> 小字保持原始大小 -> 识别率显著提高。
           各块结果合并去重。"""
        H, W = img.shape[:2]
        drop_latin = bool(kw.get('drop_latin', False))
        ov_y, ov_x = int(MAXH * 0.08), int(MAXW * 0.06)
        ys, y = [], 0
        while y < H:
            y2 = min(H, y + MAXH)
            ys.append((y, y2))
            if y2 >= H:
                break
            y = y2 - ov_y
        xs, x = [], 0
        while x < W:
            x2 = min(W, x + MAXW)
            xs.append((x, x2))
            if x2 >= W:
                break
            x = x2 - ov_x
        n_tile = len(ys) * len(xs)
        _log(f"  分块处理: {W}x{H} → {len(ys)}x{len(xs)} = {n_tile} 块"
             f" (每块 ≤{MAXW}x{MAXH}, 避免整体缩放)")
        bars, texts, bc, tc = [], [], {}, {}
        T = PL = B = O = 0.0
        bsrc = '-'
        for i, (y1, y2) in enumerate(ys, 1):
            for j, (x1, x2) in enumerate(xs, 1):
                sub = img[y1:y2, x1:x2]
                self._last_texts = []
                R, _ = self.process(sub, **kw)
                if R.get('error'):
                    continue
                T += R['total']; PL += R['pl']; B += R['t_bar']; O += R['t_ocr']
                if R.get('bar_src', '未检出') != '未检出':
                    bsrc = R['bar_src']
                for b in R['bars']:
                    key = (b[0], b[1])
                    bc[key] = bc.get(key, 0) + 1
                    if key not in [(z[0], z[1]) for z in bars]:
                        bars.append(b)
                for t in R['texts']:
                    tc[t] = tc.get(t, 0) + 1
                    if t not in texts:
                        texts.append(t)
                _log(f"    块[{i},{j}] {sub.shape[1]}x{sub.shape[0]}: "
                     f"条码 {len(R['bars'])} | 文字 {len(R['texts'])} | {R['total']:.0f}ms")
        # 排序: 校验通过的条码优先
        bars.sort(key=lambda b: (0 if self.bar_checksum(b[0], b[1]) is True else 1))
        # ⭐ 关键: 分块重叠区会产生同内容的多种识别 -> 用相似行合并 + 乱码过滤
        texts = merge_lines(texts, drop_latin=drop_latin)
        if bars:
            # 已有条码时, 丢掉条码下方那行纯数字噪声(内容已由条码给出)
            texts = [t for t in texts if not _is_barcode_noise(t)]
        texts.sort(key=lambda t: -tc.get(t, 0))
        R = {'bars': bars, 'texts': texts, 'bar_src': bsrc, 'bar_counts': bc,
             'text_counts': tc, 'nframes': n_tile, 'gate': False, 'cached': False,
             'pl': PL / max(n_tile, 1), 't_bar': B / max(n_tile, 1), 't_roi': 0,
             't_ocr': O / max(n_tile, 1), 'total': T, 'tiled': n_tile,
             'conf': round(text_score(texts), 1)}
        return R

    def _fallback_ocr(self, binary, cropmax, R, drop_latin=False):
        area = binary.shape[0] * binary.shape[1]
        if area > 1_300_000:
            _log("  (未找到文字区域, 且整幅过大不兜底; 请用 crop= 指定区域)")
            return 0.0
        w0 = binary.shape[1]
        full = binary if w0 <= cropmax else cv2.resize(
            binary, (cropmax, max(8, int(binary.shape[0] * cropmax / w0))))
        full, _ = prep_for_ocr(full, target_h=36, do_deskew=True, border=16)
        t0 = time.perf_counter()
        lines, conf, _dt = self.ocr_one(full, psm=6, dawg=True)
        R['texts'] = merge_lines(lines, drop_latin=drop_latin)
        R['fallback'] = True
        R['conf'] = round(text_score(R['texts']), 1)     # 用自算质量分, 不用不可靠的置信度 API
        self._last_texts = list(R['texts'])
        return (time.perf_counter() - t0) * 1000

    # ---------------- 裁剪 ----------------
    def auto_crop_box(self, img, margin=0.02, excl=()):
        """自动找出文字区域, 返回原图坐标下的裁剪框 (x0,y0,x1,y1)
           原理: Otsu 二值化 -> 行/列投影 -> 取墨迹密集范围
           excl: 需要排除的矩形(条码区), 避免把条码当文字裁进来"""
        h, w = img.shape[:2]
        s = min(1.0, 1200.0 / max(w, h))
        small = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s)))) if s < 1 else img
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 抹掉条码区域(否则条码会被当成文字区)
        for e in excl:
            ex0, ey0, ew, eh = [int(v * s) for v in e]
            pad = 6
            cv2.rectangle(bw, (max(0, ex0 - pad), max(0, ey0 - pad)),
                          (min(bw.shape[1], ex0 + ew + pad), min(bw.shape[0], ey0 + eh + pad)), 0, -1)
        k = max(3, int(min(small.shape[:2]) * 0.004))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        rows = bw.sum(axis=1) / 255.0
        cols = bw.sum(axis=0) / 255.0

        def span(proj):
            th = max(3.0, proj.max() * 0.08)
            idx = np.where(proj > th)[0]
            return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(proj) - 1)

        y0, y1 = span(rows)
        x0, x1 = span(cols)
        mx = int((x1 - x0) * margin)
        my = int((y1 - y0) * margin)
        bx0, by0 = max(0, int(x0 / s) - mx), max(0, int(y0 / s) - my)
        bx1, by1 = min(w, int(x1 / s) + mx), min(h, int(y1 / s) + my)
        # ⭐ 裁剪框必须包含已检测到的条码, 否则会把条码裁掉导致漏识别
        for e in excl:
            bx0 = min(bx0, max(0, int(e[0])))
            by0 = min(by0, max(0, int(e[1])))
            bx1 = max(bx1, min(w, int(e[0] + e[2])))
            by1 = max(by1, min(h, int(e[1] + e[3])))
        box = (bx0, by0, bx1, by1)
        # 裁剪框太极端就不用(可能是整页文字或误检)
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
        if area_ratio < 0.05 or area_ratio > 0.97:
            return None, area_ratio
        return box, area_ratio

    def apply_crop(self, img, crop, excl=()):
        """crop: None / 'auto' / (x0,y0,x1,y1)   excl: 排除区(条码)"""
        if crop is None or crop is False:
            return img, None
        if isinstance(crop, str) and crop.lower() == 'auto':
            box, ratio = self.auto_crop_box(img, excl=excl)
            if box is None:
                _log(f"  自动裁剪: 未找到合适区域(占比 {ratio:.0%}), 保持原图")
                return img, None
            _log(f"  自动裁剪: {box[0]},{box[1]} → {box[2]},{box[3]}"
                 f"  ({img.shape[1]}x{img.shape[0]} → {box[2]-box[0]}x{box[3]-box[1]}, 占比 {ratio:.0%})")
            return img[box[1]:box[3], box[0]:box[2]], box
        if isinstance(crop, (tuple, list)) and len(crop) == 4:
            x0, y0, x1, y1 = [int(v) for v in crop]
            h, w = img.shape[:2]
            x0, x1 = max(0, min(x0, w - 1)), max(1, min(x1, w))
            y0, y1 = max(0, min(y0, h - 1)), max(1, min(y1, h))
            _log(f"  手动裁剪: {x0},{y0} → {x1},{y1}  ({x1-x0}x{y1-y0})")
            return img[y0:y1, x0:x1], (x0, y0, x1, y1)
        _log(f"  ⚠️ crop 参数无效: {crop!r} (应为 None / 'auto' / (x0,y0,x1,y1))")
        return img, None

    # ---------------- 图片定位(找不到时给出友好提示) ----------------
    def _resolve_image(self, path):
        if path:
            for q in (path, os.path.join(OUTDIR, path), os.path.join(OUTDIR, 'docs', path)):
                if os.path.isfile(q):
                    return q
            _log(f"  ⚠️ 读不到指定的图片: {path}")
        else:
            _log("  (未指定图片, 自动选择最新的一张)")

        exts = ('jpg', 'jpeg', 'png', 'bmp', 'webp', 'JPG', 'JPEG', 'PNG', 'BMP', 'WEBP')
        cands = []
        for d in ('docs', ''):
            base = os.path.join(OUTDIR, d) if d else OUTDIR
            for e in exts:
                cands += glob.glob(os.path.join(base, f'*.{e}'))
        if not cands:
            _log(f"  ❌ {OUTDIR} 与 {OUTDIR}/docs 里都没有图片文件")
            _log(f"     请把图片上传到 docs/ 文件夹 (支持 {'/'.join(exts[:5])})")
            return None
        cands.sort(key=os.path.getmtime, reverse=True)
        _log(f"  可用图片: {', '.join(os.path.basename(c) for c in cands[:6])}")
        _log(f"  → 自动选择最新: {os.path.basename(cands[0])}")
        return cands[0]

    # ---------------- 场景 1: 单张图片 ----------------
    def image(self, path='', do_ocr=True, max_roi=3, cropmax=500, rotate='0', crop=None,
              accurate=True, min_conf=0, tile=None, drop_latin=False):
        if not _acquire_lock():
            return
        try:
            self.init()
            path = self._resolve_image(path if isinstance(path, str) else '')
            if not path:
                return
            img = cv2.imread(path)
            if img is None:
                _log(f"❌ 无法解码图片(格式不支持?): {path}")
                return
            _log(f"\n{'='*72}")
            _log(f"  图片识别: {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}")
            _log(f"{'='*72}")
            # 裁剪(在旋转前, 坐标按原图)
            if crop is not None and crop is not False:
                excl = ()
                if isinstance(crop, str) and crop.lower() == 'auto':
                    # 先快速找一下条码, 自动裁剪时把它们排除(否则条码会被当文字区送进 OCR 瞎猜)
                    try:
                        from pyzbar import pyzbar
                        _g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        _b0 = pyzbar.decode(_g0, symbols=SYMS) if SYMS else pyzbar.decode(_g0)
                        excl = tuple((c.rect.left, c.rect.top, c.rect.width, c.rect.height)
                                     for c in _b0)
                        if excl:
                            _log(f"  自动裁剪将排除 {len(excl)} 个条码区域")
                    except Exception:
                        pass
                img, box = self.apply_crop(img, crop, excl=excl)
                if box:
                    _log(f"  裁剪后尺寸: {img.shape[1]}x{img.shape[0]}")
            if max(img.shape[:2]) > MAXH and (img.shape[1] > MAXW or img.shape[0] > MAXH):
                if tile is None or tile:
                    _log(f"  ℹ️ 图幅 {img.shape[1]}x{img.shape[0]} 超过 PL 上限 {MAXW}x{MAXH}"
                         f" → 将分块处理(不缩放, 字保持原始大小)")
                else:
                    _log(f"  ⚠️ 图幅超过 PL 上限 {MAXW}x{MAXH}, 已自动缩放"
                         f"(字变小会影响识别; 可用 crop=(x0,y0,x1,y1) 或 tile=True)")
            # 方向识别(默认 auto: 用 OSD 判断; 也可直接给 0/90/180/270)
            if str(rotate).lower() == 'auto':
                rot = self.detect_rotation(img)
            else:
                rot = ROTMAP.get(str(rotate))
                _log(f"  [方向] 手动指定: {ROTNAME[rot]}")
            img = self._apply_rot(img, rot)
            self._last_texts = []
            # 大图默认分块(不再整体缩放, 保住小字), 也可用 tile=False 关闭
            if tile is None:
                tile = (getattr(self, 'ocr_layout', 'auto') == 'legacy' and
                        (img.shape[1] > MAXW or img.shape[0] > MAXH))
            if tile and (img.shape[1] > MAXW or img.shape[0] > MAXH):
                R = self._process_tiled(img, do_ocr=do_ocr, max_roi=max_roi, cropmax=cropmax,
                                        accurate=accurate, min_conf=min_conf,
                                        drop_latin=drop_latin)
            else:
                R, _ = self.process(img, do_ocr=do_ocr, max_roi=max_roi, cropmax=cropmax,
                                    accurate=accurate, min_conf=min_conf, drop_latin=drop_latin)
            if R.get('error'):
                _log(f"  ❌ {R['error']}")
                return
            _log(f"  分辨率(PL): {self.plw}x{self.plh}   阈值 {R.get('thresh')}")
            self.print_summary(R, os.path.basename(path))
            return R
        finally:
            _release_lock()

    # ---------------- 场景 2: 视频 ----------------
    def video(self, path='', per_sec=1.0, maxframes=0, gate=True, gate_th=10.0,
              do_ocr=True, max_seconds=0, max_roi=3, cropmax=450, rotate='auto',
              accurate=False, min_conf=0, step=None, tile=None, drop_latin=False):
        """按时间采样识别(默认每 1 秒 1 帧):
             视频 5 秒 -> 输出 5 次识别结果, 每次带时间戳
           per_sec=0 可改回按帧采样 step(每 N 帧取 1)"""
        if not _acquire_lock():
            return
        try:
            self.init()
            if not path:
                cands = []
                for ext in ('mp4', 'avi', 'mov', 'mkv', 'MP4', 'AVI', 'MOV'):
                    cands += glob.glob(os.path.join(OUTDIR, '*.' + ext))
                if not cands:
                    _log("❌ 目录里没有视频文件"); return
                cands.sort(key=os.path.getmtime, reverse=True)
                path = cands[0]
                _log(f"  (未指定视频, 自动选择最新: {os.path.basename(path)})")
            elif not os.path.isabs(path):
                path = os.path.join(OUTDIR, path)

            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                _log(f"❌ 无法解码视频: {path}"); return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            VW, VH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dur = total / max(fps, 1.0)
            # 采样间隔: 按时间(默认每秒 1 帧) 或 按帧
            if per_sec and per_sec > 0:
                step = max(1, int(round(fps * float(per_sec))))
                mode_txt = f"每 {per_sec:g} 秒 1 帧  (≈每 {step} 帧取 1)"
                n_expect = max(1, int(dur / float(per_sec)) + 1)
            else:
                step = int(step) if step else 4
                mode_txt = f"每 {step} 帧 1 帧"
                n_expect = max(1, total // step)
            _log(f"\n{'='*74}")
            _log(f"  视频识别: {os.path.basename(path)}  {VW}x{VH}  {fps:.1f}fps  {total}帧"
                 f" ({dur:.1f}秒)")
            _log(f"  采样: {mode_txt}   预计 {n_expect} 次识别"
                 + (f"   最多 {maxframes} 次" if maxframes else ""))
            _log(f"  门控={'开' if gate else '关'}  OCR={'开' if do_ocr else '关'}"
                 f"  高精度={'开' if accurate else '关'}")
            _log(f"{'='*74}")

            # 方向识别(用第一帧; auto=OSD 判定, 也可直接给 0/90/180/270)
            rot = None
            if str(rotate).lower() == 'auto':
                ok0, fr0 = cap.read()
                if ok0:
                    rot = self.detect_rotation(fr0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                rot = ROTMAP.get(str(rotate))
                _log(f"  [方向] 手动指定: {ROTNAME[rot]}")

            _log(f"  {'时间':>7}{'帧':>7}{'总':>9}{'PL':>7}{'条码':>8}{'OCR':>8}{'门控':>6}  识别结果")
            prev = None
            T = PL = B = O = 0.0
            bh = th = 0
            allR = []
            t_start = time.time()
            n = 0
            idx = 0
            while True:                                   # ⭐ 流式处理, 不把整段视频读进内存
                if maxframes and n >= maxframes:
                    break
                if max_seconds > 0 and (time.time() - t_start) > max_seconds:
                    _log(f"  ⏱ 达到时间上限 {max_seconds:.0f}s, 提前停止 (已处理 {n} 次)")
                    break
                ok, fr = cap.read()
                if not ok:
                    break
                if idx % step == 0:
                    ts = idx / max(fps, 1.0)
                    frame = self._apply_rot(fr, rot)
                    if tile and (frame.shape[1] > MAXW or frame.shape[0] > MAXH):
                        R = self._process_tiled(frame, do_ocr=do_ocr, max_roi=max_roi,
                                                cropmax=cropmax, accurate=accurate,
                                                min_conf=min_conf, drop_latin=drop_latin)
                        prev = None
                    else:
                        R, prev = self.process(frame, do_ocr=do_ocr,
                                               prev_thumb=prev if gate else None,
                                               gate_th=gate_th, max_roi=max_roi,
                                               cropmax=cropmax, accurate=accurate,
                                               min_conf=min_conf, drop_latin=drop_latin)
                    if R.get('error'):
                        _log(f"  {ts:6.1f}s{idx:7d}  ❌ {R['error']}")
                        idx += 1
                        continue
                    n += 1
                    allR.append(R)
                    T += R['total']; PL += R['pl']; B += R['t_bar']; O += R['t_ocr']
                    if R['bars']: bh += 1
                    if R['texts']: th += 1
                    desc = ' '.join(f"[{b[0]}]{b[1]}" for b in R['bars'][:2])
                    if R['texts']:
                        desc += (' | ' if desc else '') + ' / '.join(R['texts'][:2])
                        if R.get('cached'):
                            desc += ' (复用上次)'
                    _log(f"  {ts:6.1f}s{idx:7d}{R['total']:8.1f}{R['pl']:6.2f}"
                         f"{R['t_bar']:7.1f}{R['t_ocr']:8.1f}"
                         f"{'跳过' if R['gate'] else '执行':>6}  {desc[:50]}")
                idx += 1
            cap.release()
            if not allR:
                _log("❌ 没有取到有效帧"); return
            n = len(allR)
            _log(f"\n  ---- 统计 ----")
            _log(f"  共 {n} 次识别   平均每次 {T/n:7.1f} ms -> {1000/max(T/n,0.1):5.2f} 次/秒")
            _log(f"  条码 {bh}/{n} 次   文字 {th}/{n} 次")
            _log(f"  阶段均值: PL {PL/n:.2f} | 条码 {B/n:.1f} | OCR {O/n:.1f} ms")
            # ⭐ 多帧投票: 同一结果在多帧出现则更可信; 校验通过的条码优先
            from collections import Counter
            bcnt, tcnt, brect = Counter(), Counter(), {}
            bsrc = '-'
            for R in allR:
                for b in R['bars']:
                    bcnt[(b[0], b[1])] += 1
                    brect.setdefault((b[0], b[1]), b[2])
                    if R.get('bar_src', '未检出') != '未检出':
                        bsrc = R['bar_src']
                for t in R['texts']:
                    tcnt[t] += 1
            nf = len(allR)
            cands = [((t, d), c, self.bar_checksum(t, d), brect[(t, d)])
                     for (t, d), c in bcnt.items()]
            valid_exist = any(c[2] is True for c in cands)
            # 有校验通过的候选时, 丢掉"校验失败且只出现1次"的(典型误读)
            cands = [c for c in cands if not (valid_exist and c[2] is False and c[1] <= 1)]
            cands.sort(key=lambda c: (0 if c[2] is True else (1 if c[2] is None else 2), -c[1]))
            ub = [(c[0][0], c[0][1], c[3]) for c in cands]
            ut = merge_lines([t for t, _c in tcnt.most_common()],
                            drop_latin=drop_latin)     # 相似行合并
            if ub:
                ut = [t for t in ut if not _is_barcode_noise(t)]      # 去条码数字噪声
            conc = [R.get('conf', -1) for R in allR if R.get('conf', -1) >= 0]
            if allR:
                _log("")
                self.print_summary({'bars': ub, 'texts': ut, 'bar_src': bsrc,
                                    'bar_counts': dict(bcnt), 'text_counts': dict(tcnt),
                                    'nframes': nf,
                                    'conf': round(sum(conc) / len(conc), 1) if conc else -1,
                                    'pl': PL / n, 't_bar': B / n, 't_roi': 0, 't_ocr': O / n,
                                    'total': T / n},
                                   f"{os.path.basename(path)}  全片汇总(多帧投票)")
            return allR
        finally:
            _release_lock()

    # ---------------- 场景 3: 批量图片 ----------------
    def folder(self, dirpath='docs', exts=('jpg', 'jpeg', 'png', 'bmp'),
               do_ocr=True, max_seconds=0, max_files=0, max_roi=3, cropmax=500,
               rotate='auto', crop=None, accurate=True, min_conf=0, drop_latin=False):
        if not _acquire_lock():
            return
        try:
            self.init()
            if not os.path.isabs(dirpath):
                dirpath = os.path.join(OUTDIR, dirpath)
            files = []
            for e in exts:
                files += glob.glob(os.path.join(dirpath, f'*.{e}'))
                files += glob.glob(os.path.join(dirpath, f'*.{e.upper()}'))
            files.sort()
            if max_files:
                files = files[:max_files]
            if not files:
                _log(f"❌ {dirpath} 里没有图片 (支持 {exts})")
                _log(f"   提示: 在 Jupyter 里新建文件夹 docs/, 把图片上传进去即可")
                return
            _log(f"\n{'='*72}")
            _log(f"  批量识别: {dirpath}  共 {len(files)} 张图  OCR={'开' if do_ocr else '关'}")
            _log(f"  ⚠️ 中文密集图片很慢(单张可达 170 秒), 建议设 max_seconds 限时")
            _log(f"{'='*72}")
            T = 0.0
            bh = th = 0
            t_start = time.time()
            rot = None
            rot_done = False
            allR = []
            for i, f in enumerate(files, 1):
                if max_seconds > 0 and (time.time() - t_start) > max_seconds:
                    _log(f"  ⏱ 达到时间上限 {max_seconds:.0f}s, 已处理 {i-1}/{len(files)} 张")
                    break
                img = cv2.imread(f)
                if img is None:
                    continue
                if crop is not None and crop is not False:
                    img, _b = self.apply_crop(img, crop)
                # 方向: auto 时用第一张图判定, 之后沿用(避免每张都花 15 秒)
                if str(rotate).lower() == 'auto':
                    if not rot_done:
                        rot = self.detect_rotation(img)
                        rot_done = True
                else:
                    if not rot_done:
                        rot = ROTMAP.get(str(rotate))
                        _log(f"  [方向] 手动指定: {ROTNAME[rot]}")
                        rot_done = True
                img = self._apply_rot(img, rot)
                self._last_texts = []
                R, _ = self.process(img, do_ocr=do_ocr, max_roi=max_roi, cropmax=cropmax,
                                    accurate=accurate, min_conf=min_conf, drop_latin=drop_latin)
                if R.get('error'):
                    _log(f"  [{i}/{len(files)}] {os.path.basename(f)} ❌ {R['error']}"); continue
                allR.append(R)
                T += R['total']
                if R['bars']: bh += 1
                if R['texts']: th += 1
                desc = ' '.join(f"[{b[0]}]{b[1]}" for b in R['bars'][:2])
                if R['texts']:
                    desc += (' | ' if desc else '') + ' / '.join(R['texts'][:1])
                _log(f"  [{i}/{len(files)}] {os.path.basename(f)[:28]:<28}{R['total']:8.0f}ms  {desc[:44]}")
            done = max(1, i)
            _log(f"\n  ---- 统计 ----")
            _log(f"  已处理 {done} 张, 平均 {T/done:.0f} ms/张")
            _log(f"  条码 {bh}/{done}   文字 {th}/{done}")
            # 汇总(统计出现张数, 校验通过的条码优先)
            from collections import Counter
            bcnt, tcnt, brect = Counter(), Counter(), {}
            bsrc = '-'
            for R in allR:
                for b in R['bars']:
                    bcnt[(b[0], b[1])] += 1
                    brect.setdefault((b[0], b[1]), b[2])
                    if R.get('bar_src', '未检出') != '未检出':
                        bsrc = R['bar_src']
                for t in R['texts']:
                    tcnt[t] += 1
            cands = [((t, d), c, self.bar_checksum(t, d), brect[(t, d)])
                     for (t, d), c in bcnt.items()]
            cands.sort(key=lambda c: (0 if c[2] is True else (1 if c[2] is None else 2), -c[1]))
            ub = [(c[0][0], c[0][1], c[3]) for c in cands]
            if allR:
                _log("")
                self.print_summary({'bars': ub, 'texts': [t for t, _ in tcnt.most_common()],
                                    'bar_src': bsrc, 'bar_counts': dict(bcnt),
                                    'text_counts': dict(tcnt), 'nframes': len(allR),
                                    'pl': 0, 't_bar': 0, 't_roi': 0, 't_ocr': 0,
                                    'total': T / done},
                                   f"{os.path.basename(dirpath)}  共 {len(allR)} 张图(汇总)")
            return allR
        finally:
            _release_lock()

    # ---------------- 工具 ----------------
    def reset(self):
        """硬件状态异常时调用, 重新加载 Overlay"""
        _log("重置中 ...")
        self.ready = False
        self.init(force=True)
        _log("完成")


# ============================================================
#  全局单例 + 便捷函数(Notebook 里直接用这些)
# ============================================================
_T = Terminal()


def setup(ocr_lang='chi_sim', morph_mode=1, force=False):
    """初始化并选择 OCR 语言 / 形态学模式 (不调用也行, run_* 会自动初始化)"""
    _log(f"pynq_ocr 模块版本: {VERSION}")
    if force:
        _T.ready = False
    _T.morph_mode = morph_mode
    _T.init(ocr_lang=ocr_lang, morph_mode=morph_mode, force=force)
    _T.status()


def set_morph_mode(mode):
    """形态学模式: 1=腐蚀(加粗黑字, 默认)  0=直通(保留原样)  2=膨胀"""
    _T.morph_mode = int(mode)
    _log(f"形态学模式已设为 {_T.morph_mode}  (1=腐蚀 0=直通 2=膨胀)")


def set_lang(ocr_lang):
    """切换 OCR 语言: 'chi_sim'(中文,默认) / 'eng'(英文,快 4 倍)。⚠️ 'chi_sim+eng' 会让中文被强行识别成拉丁乱码，不要用于中文场景"""
    _T.ready = False
    _T.init(ocr_lang=ocr_lang, morph_mode=_T.morph_mode, force=True)


def status():
    _T.status()


def reset():
    _T.reset()


def run_image(path='', **kw):
    """单张图片识别:  run_image('photo.jpg')  或  run_image() 自动选最新一张
       裁剪参数: crop='auto' 自动裁到文字区域 / crop=(x0,y0,x1,y1) 手动指定"""
    return _T.image(path, **kw)


def preview(path='', crop=None, grid=True):
    """在 Notebook 里显示图片(带坐标网格), 用来确定 crop 的坐标"""
    try:
        from IPython.display import display
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        _log("预览需要 matplotlib, 改为只显示尺寸与建议裁剪框")
        matplotlib = None
    path = _T._resolve_image(path if isinstance(path, str) else '')
    if not path:
        return
    img = cv2.imread(path)
    if img is None:
        _log(f"❌ 无法读取图片: {path}")
        return
    h, w = img.shape[:2]
    _log(f"图片: {os.path.basename(path)}  {w}x{h} 像素")
    box, ratio = _T.auto_crop_box(img)
    if box:
        _log(f"自动检测的文字区域: crop=({box[0]}, {box[1]}, {box[2]}, {box[3]})  占比 {ratio:.0%}")
    else:
        _log(f"未自动检测到明显文字区域(占比 {ratio:.0%})")
    if crop:
        img, _ = _T.apply_crop(img, crop)
    try:
        from IPython.display import display
        rgb = cv2.cvtColor(cv2.resize(img, (min(700, img.shape[1]),
                                            int(img.shape[0] * min(700, img.shape[1]) / img.shape[1]))),
                           cv2.COLOR_BGR2RGB)
        _, ax = plt.subplots(figsize=(9, 9 * rgb.shape[0] / rgb.shape[1]))
        ax.imshow(rgb)
        if grid:
            hh, ww = rgb.shape[:2]
            step = max(50, ww // 8)
            ax.set_xticks(range(0, ww, step)); ax.set_yticks(range(0, hh, step))
            ax.grid(color='r', alpha=0.35, linewidth=0.5)
            ax.set_xlabel(f"显示宽度 {ww}px (原图 {img.shape[1]}px, 比例 {img.shape[1]/ww:.2f}x)")
        plt.tight_layout(); plt.show()
    except Exception as e:
        _log(f"(图形显示不可用: {e})")


def crop_save(path='', box=None, out=''):
    """把裁剪结果另存为新文件, 方便以后直接识别"""
    path = _T._resolve_image(path if isinstance(path, str) else '')
    if not path:
        return
    img = cv2.imread(path)
    if img is None:
        _log(f"❌ 无法读取图片: {path}")
        return
    if box is None:
        box, ratio = _T.auto_crop_box(img)
        if box is None:
            _log(f"❌ 未检测到文字区域(占比 {ratio:.0%}), 请手动给 box=(x0,y0,x1,y1)")
            return
    _T.init()
    img2, _ = _T.apply_crop(img, box)
    if not out:
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(OUTDIR, base + f'_crop{img2.shape[1]}x{img2.shape[0]}.jpg')
    cv2.imwrite(out, img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    _log(f"已保存: {out}  ({img2.shape[1]}x{img2.shape[0]})")


def run_video(path='', **kw):
    """视频识别(默认每 1 秒识别 1 次, 输出每次的结果):
         run_video()                            自动选最新视频, 每秒 1 次
         run_video(per_sec=0.5)                 每 0.5 秒 1 次
         run_video(per_sec=0, step=15)          改回按帧采样(每 15 帧 1 次)
         run_video(do_ocr=False)                只做条码(最快)
         run_video(max_seconds=120)             最多跑 120 秒"""
    return _T.video(path, **kw)


def run_folder(dirpath='docs', **kw):
    """批量图片:  run_folder('docs', max_seconds=120)"""
    return _T.folder(dirpath, **kw)


def list_files():
    """看看板卡目录里有哪些素材"""
    _log(f"目录: {OUTDIR}")
    for ptn in ('*.mp4', '*.avi', '*.mov', '*.jpg', '*.jpeg', '*.png'):
        for f in sorted(glob.glob(os.path.join(OUTDIR, ptn))):
            _log(f"  {os.path.basename(f):<34}{os.path.getsize(f)/1024:>9.1f} KB")
    for d in ('docs', 'tessdata_fast'):
        p = os.path.join(OUTDIR, d)
        if os.path.isdir(p):
            n = len([x for x in os.listdir(p) if not x.startswith('.')])
            _log(f"  [目录] {d}/  ({n} 个文件)")
