"""Local PS OCR pipeline; no new dependencies, PL interface unchanged.

Tesseract C API: https://github.com/tesseract-ocr/tesseract/blob/4.1.1/src/api/capi.h
"""
import copy
import csv
import ctypes as C
import io
import math
import time

import cv2
import numpy as np


def bind_api(lib):
    """Bind pointers AND return types; MeanTextConf returns int, not float."""
    p, i, s = C.c_void_p, C.c_int, C.c_char_p
    signatures = {
        'TessBaseAPICreate': ([], p),
        'TessBaseAPIDelete': ([p], None),
        'TessBaseAPIInit3': ([p, s, s], i),
        'TessBaseAPISetVariable': ([p, s, s], i),
        'TessBaseAPISetPageSegMode': ([p, i], None),
        'TessBaseAPISetSourceResolution': ([p, i], None),
        'TessBaseAPISetImage': ([p, s, i, i, i, i], None),
        'TessBaseAPIRecognize': ([p, p], i),
        'TessBaseAPIGetUTF8Text': ([p], p),
        'TessBaseAPIGetTsvText': ([p, i], p),
        'TessBaseAPIMeanTextConf': ([p], i),
        'TessBaseAPIClear': ([p], None),
        'TessBaseAPIEnd': ([p], None),
        'TessDeleteText': ([p], None),
        'TessMonitorCreate': ([], p),
        'TessMonitorDelete': ([p], None),
        'TessMonitorSetDeadlineMSecs': ([p, i], None),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)  # fail clearly for incompatible libraries
        fn.argtypes, fn.restype = args, result
    return lib


def detect_lines(gray, max_width=1600, min_contrast=18):
    """Scale-aware character components -> horizontal lines, without top-4 cap.

    Adaptive threshold is used only for detection. Recognition retains grayscale.
    Returns original-image rectangles, estimated character height, component count.
    """
    h0, w0 = gray.shape
    scale = min(1.0, max_width / float(w0))
    g = cv2.resize(gray, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
                   interpolation=cv2.INTER_AREA) if scale < 1 else gray
    h, w = g.shape
    if min(h, w) < 8:
        return [], 12.0, 0
    block = min(41, min(h, w) // 2 * 2 - 1)
    mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, max(3, block), 12)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    x, y, cw, ch, area = stats.T
    valid = ((area >= 5) & (ch >= 5) & (ch <= max(20, h * .35)) &
             (cw >= 1) & (cw <= max(30, w * .12)) &
             (cw / np.maximum(ch, 1) < 5) &
             (area / np.maximum(cw * ch, 1) < .94))
    valid[0] = False
    heights = ch[valid]
    char_h = float(np.median(heights)) if heights.size else 12 * scale
    clean = np.where(valid[labels], np.uint8(255), np.uint8(0))
    # Horizontal joining only: vertical dilation can glue dense adjacent lines.
    joined = cv2.dilate(clean, np.ones((1, max(3, round(char_h * 1.2))), np.uint8))
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    selected_heights = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        if bw < max(12, char_h * 1.3) or bh < 5 or bh > max(40, char_h * 3.5):
            continue
        if np.count_nonzero(clean[by:by + bh, bx:bx + bw]) < 12:
            continue
        # Dark-object boundaries under vignetting form low-contrast pseudo-lines.
        if float(g[by:by + bh, bx:bx + bw].std()) < min_contrast:
            continue
        ids = np.unique(labels[by:by + bh, bx:bx + bw])
        selected_heights.extend(ch[ids[valid[ids]]].tolist())
        pad = max(2, round(char_h * .18))
        x0, y0 = max(0, bx - pad), max(0, by - pad)
        x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
        boxes.append((round(x0 / scale), round(y0 / scale),
                      round((x1 - x0) / scale), round((y1 - y0) / scale)))
    # Spaces and mixed font sizes can split one physical line into several
    # contours. Rejoin nearby boxes on the SAME baseline, preserving columns.
    merged = []
    for rect in sorted(boxes, key=lambda r: (r[1] + r[3] / 2, r[0])):
        x, y, bw, bh = rect
        for index, (ax, ay, aw, ah) in enumerate(merged):
            overlap = max(0, min(y + bh, ay + ah) - max(y, ay))
            gap = max(0, max(x, ax) - min(x + bw, ax + aw))
            if overlap >= .6 * min(bh, ah) and gap <= .9 * max(bh, ah) and .4 <= bh / ah <= 2.5:
                x0, y0 = min(x, ax), min(y, ay)
                merged[index] = (x0, y0, max(x+bw, ax+aw)-x0, max(y+bh, ay+ah)-y0)
                break
        else:
            merged.append(rect)
    boxes = sorted(merged, key=lambda r: (r[1], r[0]))
    if selected_heights:
        char_h = float(np.median(selected_heights))
    return boxes, char_h / scale, int(valid.sum())


def _cjk(ch):
    return '\u3400' <= ch <= '\u9fff'


TSV_HEADER = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t'
              'left\ttop\twidth\theight\tconf\ttext')


def parse_tsv(tsv):
    """One result per physical line, including repeated and one-character lines.

    注意: Tesseract 的 C API TessBaseAPIGetTsvText() **不输出表头行**
    (只有命令行 tesseract 的 tsv 输出才带表头)。若把数据直接交给
    csv.DictReader, 第一行数据会被当成表头, 导致 row['level'] 恒为 None,
    最终解析出 0 行文字(实板表现: conf 正常但 texts 全空)。
    这里判断首行是否以 'level' 开头, 缺表头就补上。
    """
    if not tsv:
        return []
    tsv = tsv.lstrip('\ufeff').lstrip('\r\n')
    if not tsv.split('\n', 1)[0].startswith('level'):
        tsv = TSV_HEADER + '\n' + tsv
    groups = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row.get('level') != '5' or not row.get('text', '').strip():
            continue
        try:
            x, y, w, h = (int(row[k]) for k in ('left', 'top', 'width', 'height'))
            conf = float(row['conf'])
            key = tuple(row[k] for k in ('page_num', 'block_num', 'par_num', 'line_num'))
        except (KeyError, ValueError):
            continue
        groups.setdefault(key, []).append((row['text'].strip(), x, y, w, h, conf))
    out = []
    for words in groups.values():
        text = ''
        for word, *_ in words:
            sep = ' ' if text and not (_cjk(text[-1]) or _cjk(word[0])) else ''
            text += sep + word
        x0, y0 = min(t[1] for t in words), min(t[2] for t in words)
        x1 = max(t[1] + t[3] for t in words)
        y1 = max(t[2] + t[4] for t in words)
        weight = sum(len(t[0]) for t in words)
        conf = sum(t[5] * len(t[0]) for t in words) / max(weight, 1)
        out.append({'text': text, 'rect': (x0, y0, x1 - x0, y1 - y0),
                    'confidence': round(conf, 1)})
    # TSV order preserves Tesseract's column/block reading order.
    return out


def recognize(terminal, image, psm, timeout_ms):
    """One recognition; TSV reuses the recognized page, no second OCR pass.

    Deadline is cooperative inside Tesseract, not a hard process watchdog.
    A failed/timed-out recognition must not call GetText (which could retry it).
    """
    lib, handle = terminal.ocr_lib, terminal.ocr
    g = np.ascontiguousarray(image, dtype=np.uint8)
    data = g.tobytes()  # retain the buffer until recognition/TSV extraction finishes
    monitor = lib.TessMonitorCreate()
    if not monitor:
        raise RuntimeError('Tesseract monitor allocation failed')
    started = time.perf_counter()
    try:
        lib.TessBaseAPISetPageSegMode(handle, int(psm))
        lib.TessBaseAPISetVariable(handle, b'tessedit_char_whitelist', b'')
        lib.TessBaseAPISetImage(handle, data, g.shape[1], g.shape[0], 1, g.shape[1])
        lib.TessBaseAPISetSourceResolution(handle, 300)
        lib.TessMonitorSetDeadlineMSecs(monitor, max(1, int(timeout_ms)))
        rc = lib.TessBaseAPIRecognize(handle, monitor)
        if rc != 0:
            return [], -1, (time.perf_counter() - started) * 1000, False
        ptr = lib.TessBaseAPIGetTsvText(handle, 0)
        if not ptr:
            return [], -1, (time.perf_counter() - started) * 1000, False
        try:
            lines = parse_tsv(C.string_at(ptr).decode('utf-8', 'replace'))
        finally:
            lib.TessDeleteText(ptr)
        conf = int(lib.TessBaseAPIMeanTextConf(handle))
        return lines, conf, (time.perf_counter() - started) * 1000, True
    finally:
        lib.TessMonitorDelete(monitor)
        lib.TessBaseAPIClear(handle)


def pack_regions(gray, boxes, char_h, max_pixels=2400000):
    """Pack all selected lines into one grayscale page and retain affine mappings."""
    scale = min(2.0, max(1.0, 28.0 / max(char_h, 1)))
    margin, gap = 12, 16
    estimated = (max(r[2] for r in boxes) * scale + margin * 2) * (
        sum(r[3] * scale + gap for r in boxes) + margin * 2)
    scale = min(scale, math.sqrt(max_pixels / max(estimated, 1)) * scale)
    pieces = []
    yy = margin
    for x, y, w, h in boxes:
        crop = gray[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        piece = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        pieces.append((piece, (x, y, w, h), yy))
        yy += nh + gap
    if not pieces:
        raise ValueError('No valid text crops')
    canvas = np.full((yy + margin, max(p[0].shape[1] for p in pieces) + margin * 2), 255, np.uint8)
    mapping = []
    for piece, original, yy in pieces:
        ph, pw = piece.shape
        canvas[yy:yy + ph, margin:margin + pw] = piece
        mapping.append(((margin, yy, pw, ph), original))
    return canvas, mapping


def restore_rect(rect, mapping, image_shape):
    x, y, w, h = rect
    cx, cy = x + w / 2, y + h / 2
    for (px, py, pw, ph), (ox, oy, ow, oh) in mapping:
        if px <= cx <= px + pw and py <= cy <= py + ph:
            x0 = max(ox, ox + (x - px) * ow / pw)
            y0 = max(oy, oy + (y - py) * oh / ph)
            x1 = min(ox + ow, ox + (x + w - px) * ow / pw)
            y1 = min(oy + oh, oy + (y + h - py) * oh / ph)
            ih, iw = image_shape[:2]
            x0, y0 = max(0, min(iw - 1, round(x0))), max(0, min(ih - 1, round(y0)))
            x1, y1 = min(iw, round(x1)), min(ih, round(y1))
            return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
    return None


def stable_frame(current, reference):
    if reference is None or current.shape != reference.shape:
        return False
    delta = cv2.absdiff(current, reference)
    # Local changes must not disappear inside a full-frame mean.
    tile_means = delta.reshape(12, 16, 16, 16).mean(axis=(1, 3))
    return float(delta.mean()) < 1.2 and float(tile_means.max()) < 4.0 and float((delta > 12).mean()) < .006


def process(terminal, image, prev_thumb=None, accurate=False, drop_latin=False,
            min_conf=0, **unused):
    started = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    thumb = cv2.resize(gray, (256, 192), interpolation=cv2.INTER_AREA)
    layout = getattr(terminal, 'ocr_layout', 'auto')
    key = (gray.shape, layout, accurate, drop_latin, min_conf,
           getattr(terminal, 'ocr_lang', ''), getattr(terminal, 'morph_mode', 1),
           unused.get('strict_text', True), getattr(terminal, 'ocr_backend', 'local'),
           getattr(terminal, 'ocr_workers', 2), getattr(terminal, 'pc_host', ''),
           getattr(terminal, 'pc_port', 9989), getattr(terminal, 'ocr_budget_ms', 0))
    cache = getattr(terminal, '_smart_cache', None)
    # Compare to the recognized frame, not the immediately preceding frame:
    # otherwise slow camera drift could reuse old text forever.
    if (prev_thumb is not None and cache and cache['key'] == key and
            time.monotonic() - cache['time'] < 2.0 and stable_frame(thumb, cache['thumb'])):
        result = copy.deepcopy(cache['result'])
        result.update(cached=True, gate=True, pl=0.0, pack=0.0, read=0.0,
                      t_bar=0.0, t_roi=0.0, t_ocr=0.0, ocr_calls=0,
                      parallel={'jobs': 0, 'workers': 0, 'split_axis': None}, pc_timing=None,
                      total=(time.perf_counter() - started) * 1000)
        terminal.images += 1
        terminal.total_ms += result['total']
        return result, thumb
    result = dict(bars=[], texts=[], rois=[], roi_texts=[], gate=False, cached=False,
                  conf=-1, t_bar=0.0, t_roi=0.0, t_ocr=0.0, ocr_calls=0,
                  complete=True, warnings=[], bar_src='未检出')
    binary, timing = terminal.preprocess(gray)
    if binary is None:
        result['error'] = '硬件事务超时'
        terminal._smart_cache = None
        return result, thumb
    result.update(timing)
    if getattr(terminal, 'ocr_backend', 'local') == 'pc':
        from pc_bridge import recognize as pc_recognize
        start_remote = time.perf_counter()
        try:
            remote_lines, details = pc_recognize(terminal, gray)
            # Keep geometry and full physical lines from neural full-frame OCR.
            for line in remote_lines:
                text = line['text'].strip()
                if text and line['confidence'] >= min_conf and (not drop_latin or any(_cjk(c) for c in text)):
                    result['texts'].append(text)
                    result['rois'].append(line['rect'])
                    result['roi_texts'].append([text])
            result['conf'] = sum(r['confidence'] for r in remote_lines) / max(1, len(remote_lines))
            result['pc_timing'] = {k: details[k] for k in ('roundtrip_ms', 'inference_ms', 'server_ms') if k in details}
            result['neural_engine'] = details.get('engine')
            result['complete'] = bool(details.get('complete', False))
        except Exception as error:
            result['complete'] = False
            result['error'] = f'PC 协助识别失败（未静默回退）: {error}'
            terminal._smart_cache = None
        result['t_ocr'] = (time.perf_counter()-start_remote)*1000
        # Barcode decode is deliberately kept on the PS. Normal scene mode
        # continues to decode all requested symbols; document layout skips it.
        if layout != 'document':
            result['bars'], result['t_bar'], result['bar_src'] = terminal.barcodes(gray, binary, accurate=False)
        result['layout'] = layout
        result['barcode_checked'] = layout != 'document'
        result['roi_src'] = 'pc-fullframe'
        result['ocr_calls'] = 1
        result['nroi'] = len(result['rois'])
        result['total'] = (time.perf_counter()-started)*1000
        terminal.images += 1
        terminal.total_ms += result['total']
        terminal._smart_cache = ({'key': key, 'thumb': thumb.copy(), 'time': time.monotonic(),
                                  'result': copy.deepcopy(result)} if result['complete'] and result['texts'] else None)
        return result, thumb
    roi_start = time.perf_counter()
    boxes, char_h, components = detect_lines(gray, min_contrast=18 if unused.get('strict_text', True) else 8)
    document = layout == 'document' or (layout == 'auto' and len(boxes) >= 6 and components >= 45)
    # Dense pages rarely need four expensive barcode decodes; retain one original
    # decode via barcodes(...accurate=False) for scene mode only. Explicit barcode
    # mode still uses the original complete barcode path.
    if not document:
        result['bars'], result['t_bar'], result['bar_src'] = terminal.barcodes(gray, binary, accurate=False)
    result['layout'] = 'document' if document else 'scene'
    result['barcode_checked'] = not document
    result['detected_lines'] = len(boxes)
    result['t_roi'] = (time.perf_counter() - roi_start) * 1000 - result['t_bar']
    if document or not boxes:
        # Keep the whole page including columns, rather than cropping the four
        # largest lines. Tesseract does the page segmentation in the same call.
        scale = min(2.0, max(1.0, 26.0 / max(char_h, 1)), math.sqrt(2400000 / gray.size))
        nw, nh = max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale))
        page = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        page = cv2.copyMakeBorder(page, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        mapping = [((12, 12, nw, nh), (0, 0, gray.shape[1], gray.shape[0]))]
        psm = 3 if document else 11
        if char_h * scale < 14:
            result['warnings'].append('文字像素较小；请靠近拍摄或使用局部裁剪')
    else:
        # Exclude decoded barcode graphics from the text canvas.
        def overlaps_bar(r):
            x, y, w, h = r
            for _, _, (bx, by, bw, bh) in result['bars']:
                overlap = max(0, min(x+w, bx+bw)-max(x, bx)) * max(0, min(y+h, by+bh)-max(y, by))
                if overlap > .5 * w * h:
                    return True
            return False
        boxes = [r for r in boxes if not overlaps_bar(r)]
        if not boxes:
            boxes = [(0, 0, gray.shape[1], gray.shape[0])]
        page, mapping = pack_regions(gray, boxes, char_h)
        psm = 6
    result['roi_src'] = 'document-gray' if document else 'line-canvas-gray'
    result['ocr_input_size'] = [page.shape[1], page.shape[0]]
    budget_ms = int(getattr(terminal, 'ocr_budget_ms', 0) or (30000 if accurate else 10000))
    ocr_start = time.perf_counter()
    from performance_runtime import recognize_batch
    if len(mapping) == 1 and document:
        (px, py, pw, ph), (_, _, ow, oh) = mapping[0]
        regions = [(round(px+x*pw/ow), round(py+y*ph/oh), round(w*pw/ow), round(h*ph/oh))
                   for x, y, w, h in boxes]
    else:
        regions = [r[0] for r in mapping]
    lines, conf, elapsed, complete, parallel = recognize_batch(terminal, page, psm, budget_ms, regions, recognize)
    result['parallel'] = parallel
    result['ocr_calls'] = parallel['jobs']
    if getattr(terminal, '_runtime_warning', ''):
        result['warnings'].append(terminal._runtime_warning)
    # Only high-accuracy mode may retry once, sharing the ORIGINAL time budget.
    # Grayscale first avoids losing fine strokes to global PL threshold/erosion.
    remaining = budget_ms - (time.perf_counter() - ocr_start) * 1000
    if accurate and complete and (not lines or conf < 45) and remaining > 500:
        alternate = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 41, 12)
        other, oc, _, done = recognize(terminal, alternate, psm, int(remaining))
        result['ocr_calls'] += 1
        if done and other and (not lines or oc > conf + 3):
            lines, conf = other, oc
            result['roi_src'] += '+adaptive'
        # A failed optional retry does not invalidate the complete first pass.
        if not done:
            result['warnings'].append('增强重试未完成，保留首轮完整结果')
    result['t_ocr'] = (time.perf_counter() - ocr_start) * 1000
    result['complete'] = complete
    if not complete:
        result['warnings'].append('OCR 超时或引擎失败；本帧未完成，请裁剪或提高 ocr_budget_ms')
    for line in lines:
        text = line['text'].strip()
        confidence_floor = max(min_conf, 20 if not document else 0)
        if not any(c.isalnum() for c in text) or line['confidence'] < confidence_floor:
            continue
        if drop_latin and not any(_cjk(c) for c in text):
            continue
        rect = restore_rect(line['rect'], mapping, gray.shape)
        if rect is None:
            continue
        result['rois'].append(rect)
        result['roi_texts'].append([text])
        result['texts'].append(text)  # no fuzzy dedup, no 12-line truncation
    result['conf'] = conf
    result['nroi'] = len(result['rois'])
    result['total'] = (time.perf_counter() - started) * 1000
    terminal.images += 1
    terminal.total_ms += result['total']
    terminal._last_texts = list(result['texts'])
    terminal._smart_cache = ({'key': key, 'thumb': thumb.copy(), 'time': time.monotonic(),
                              'result': copy.deepcopy(result)} if complete and result['texts'] else None)
    return result, thumb
