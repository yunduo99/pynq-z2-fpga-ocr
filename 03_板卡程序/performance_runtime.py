"""Two independent PS OCR handles and measurement helpers. No board deployment."""
from concurrent.futures import ThreadPoolExecutor, wait
import copy
import ctypes
import os
import threading
import time
from types import SimpleNamespace

import numpy as np


def memory_available_mb():
    try:
        with open('/proc/meminfo', encoding='ascii') as source:
            for line in source:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def close_runtime(terminal):
    runtime = getattr(terminal, '_ocr_runtime', None)
    if runtime is not None:
        runtime.close()
        terminal._ocr_runtime = None


class DualOCR:
    """One handle per concurrent job; never share a TessBaseAPI across jobs.

    Main service serializes frames/settings. PL/DMA is never called by workers.
    """
    def __init__(self, terminal):
        self.lib = terminal.ocr_lib
        self.key = (terminal.ocr, terminal.ocr_lang, terminal.tessdata_path)
        self.handles = [terminal.ocr]
        self.lock = threading.Lock()
        handle = self.lib.TessBaseAPICreate()
        if not handle:
            raise RuntimeError('Second OCR handle allocation failed')
        try:
            rc = self.lib.TessBaseAPIInit3(handle, terminal.tessdata_path.encode(), terminal.ocr_lang.encode())
            if rc != 0:
                raise RuntimeError('Second OCR model initialization failed: ' + str(rc))
            self.handles.append(handle)
            self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='ps-ocr')
        except Exception:
            self.lib.TessBaseAPIDelete(handle)
            raise

    def run(self, index, image, psm, deadline, recognizer):
        context = SimpleNamespace(ocr_lib=self.lib, ocr=self.handles[index])
        remaining = int((deadline - time.perf_counter()) * 1000)
        if remaining <= 0:
            return [], -1, 0.0, False
        return recognizer(context, image, psm, remaining)

    def close(self):
        with self.lock:
            self.pool.shutdown(wait=True)
            # The main Terminal still owns handles[0].
            for handle in self.handles[1:]:
                self.lib.TessBaseAPIDelete(handle)
            self.handles = self.handles[:1]


def prepare_runtime(terminal):
    requested = int(getattr(terminal, 'ocr_workers', 2))
    key = (terminal.ocr, terminal.ocr_lang, terminal.tessdata_path)
    current = getattr(terminal, '_ocr_runtime', None)
    if current and requested == 2 and current.key == key:
        return current
    close_runtime(terminal)
    terminal._runtime_warning = ''
    if requested != 2 or (os.cpu_count() or 1) < 2:
        return None
    available = memory_available_mb()
    if available is not None and available < 160:
        terminal._runtime_warning = '可用内存不足160MB，保留单 OCR 引擎'
        return None
    try:
        terminal._ocr_runtime = DualOCR(terminal)
    except Exception as error:
        terminal._runtime_warning = f'双引擎初始化失败，保留单引擎: {error}'
        terminal._ocr_runtime = None
    return terminal._ocr_runtime


def safe_split(shape, regions):
    """Find an internal gap not crossing ANY detected line. Prefer column gap.

    Returns (axis, coordinate), or None when layout cannot safely be split.
    Detection is approximate, so --ocr-workers 1 remains available for A/B tests.
    """
    if len(regions) < 2:
        return None
    for axis, dimension in ((1, shape[1]), (0, shape[0])):
        intervals = sorted((r[axis ^ 1], r[axis ^ 1] + r[(axis ^ 1) + 2]) for r in regions)
        merged = []
        for left, right in intervals:
            if merged and left <= merged[-1][1] + 2:
                merged[-1] = (merged[-1][0], max(merged[-1][1], right))
            else:
                merged.append((left, right))
        candidates = [(a[1]+b[0]) // 2 for a, b in zip(merged, merged[1:]) if b[0]-a[1] >= 4]
        candidates = [cut for cut in candidates if .25 * dimension < cut < .75 * dimension]
        if candidates:
            return axis, min(candidates, key=lambda cut: abs(cut - dimension / 2))
    return None


def recognize_batch(terminal, page, psm, budget_ms, regions, recognizer):
    runtime = getattr(terminal, '_ocr_runtime', None)
    split = safe_split(page.shape, regions) if runtime else None
    if split is None:
        result = recognizer(terminal, page, psm, budget_ms)
        return (*result, {'jobs': 1, 'workers': 1, 'split_axis': None})
    axis, cut = split
    parts = [(page[:, :cut], 0, 0), (page[:, cut:], cut, 0)] if axis == 1 else [
        (page[:cut], 0, 0), (page[cut:], 0, cut)]
    started = time.perf_counter()
    deadline = started + budget_ms / 1000
    with runtime.lock:
        futures = [runtime.pool.submit(runtime.run, i, np.ascontiguousarray(image), psm, deadline, recognizer)
                   for i, (image, _, _) in enumerate(parts)]
        # Wait for BOTH jobs before returning/allowing settings changes. The native
        # deadlines are cooperative; a Python timeout must not abandon busy handles.
        wait(futures)
        answers = [future.result() for future in futures]
    lines, complete, weighted_conf, count = [], True, 0.0, 0
    for (records, conf, _, done), (_, ox, oy) in zip(answers, parts):
        complete = complete and done
        for record in records:
            record = copy.deepcopy(record)
            x, y, w, h = record['rect']
            record['rect'] = (x+ox, y+oy, w, h)
            lines.append(record)
        weight = sum(len(record['text']) for record in records)
        weighted_conf += max(0, conf) * weight
        count += weight
    return lines, weighted_conf / max(count, 1), (time.perf_counter()-started)*1000, complete, {
        'jobs': 2, 'workers': 2, 'split_axis': 'columns' if axis == 1 else 'rows'}


def profile_result(result, wall_start, cpu_start, target_ms):
    elapsed = (time.perf_counter() - wall_start) * 1000
    cpu_ms = (time.process_time() - cpu_start) * 1000
    cpu_count = os.cpu_count() or 1
    result['profile'] = {'wall_ms': round(elapsed, 2), 'process_cpu_ms': round(cpu_ms, 2),
                         'average_busy_cores': round(cpu_ms/max(elapsed, .001), 3),
                         'process_share_of_all_cpus_pct': round(100*cpu_ms/max(elapsed*cpu_count, .001), 2),
                         'cpu_count': cpu_count, 'mem_available_mb': memory_available_mb()}
    result['target_ms'] = target_ms
    # Reusing old text is not a fresh recognition under the one-second target.
    result['fresh_target_met'] = bool(result.get('complete', True) and not result.get('error')
                                     and not result.get('cached') and (result.get('texts') or result.get('bars'))
                                     and elapsed <= target_ms)
    return result
