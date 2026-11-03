# -*- coding: utf-8 -*-
"""零依赖 PGM(P5) 比对工具 —— 比对 HLS 仿真输出 与 OpenCV 参考图

用法:
  python compare_pgm_pure.py                     # 比对 data/ 下所有 out_* vs ref_*
  python compare_pgm_pure.py a.pgm b.pgm
"""
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

PAIRS = [
    ('out_bin.pgm',    'ref_bin.pgm',    '二值化 (binarize)'),
    ('out_erode.pgm',  'ref_erode.pgm',  '腐蚀 (erode)'),
    ('out_dilate.pgm', 'ref_dilate.pgm', '膨胀 (dilate)'),
]


def read_pgm(path):
    with open(path, 'rb') as f:
        raw = f.read()
    # 解析 P5 头: P5 <w> <h> <maxval>\n
    tokens = []
    i = 0
    while len(tokens) < 4 and i < len(raw):
        # 跳过空白/注释
        while i < len(raw) and raw[i:i+1].isspace():
            i += 1
        if raw[i:i+1] == b'#':
            while i < len(raw) and raw[i:i+1] != b'\n':
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j:j+1].isspace():
            j += 1
        tokens.append(raw[i:j])
        i = j
    i += 1  # 头部结束的单个空白
    magic, w, h, maxv = tokens[0].decode(), int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic != 'P5':
        raise ValueError(f'{path}: 不是 P5 PGM')
    data = raw[i:i + w * h]
    return w, h, data


def compare(a_path, b_path, label):
    if not os.path.exists(a_path):
        print(f"[跳过] 缺少 HLS 输出: {os.path.basename(a_path)}")
        return None
    if not os.path.exists(b_path):
        print(f"[跳过] 缺少参考图: {os.path.basename(b_path)}")
        return None
    wa, ha, da = read_pgm(a_path)
    wb, hb, db = read_pgm(b_path)
    if (wa, ha) != (wb, hb):
        print(f"[失败] {label}: 尺寸不一致 {wa}x{ha} vs {wb}x{hb}")
        return False
    total = wa * ha
    bad = 0
    maxdiff = 0
    first = None
    for k in range(total):
        d = abs(da[k] - db[k])
        if d:
            bad += 1
            if d > maxdiff:
                maxdiff = d
            if first is None:
                first = (k % wa, k // wa, da[k], db[k])
    print(f"\n=== {label} ===")
    print(f"  尺寸       : {wa}x{ha}")
    print(f"  不一致像素 : {bad} / {total}  ({bad / total * 100:.4f}%)")
    print(f"  最大误差   : {maxdiff}")
    if bad == 0:
        print("  结论       : [PASS] 与 OpenCV 逐像素完全一致")
        return True
    print(f"  首个差异   : (x={first[0]}, y={first[1]})  HLS={first[2]}  OpenCV={first[3]}")
    print("  结论       : [FAIL] 存在差异")
    return False


if len(sys.argv) == 3:
    compare(sys.argv[1], sys.argv[2], '自定义比对')
else:
    results = []
    for a, b, label in PAIRS:
        r = compare(os.path.join(DATA, a), os.path.join(DATA, b), label)
        if r is not None:
            results.append((label, r))
    print("\n" + "=" * 50)
    print("汇总:")
    for label, ok in results:
        print(f"  {label:22s}: {'[PASS]' if ok else '[FAIL]'}")
    if results and all(r for _, r in results):
        print("\n>>> 全部通过: HLS 算子输出与 OpenCV 参考完全一致 <<<")
    elif results:
        print("\n>>> 存在未通过项 <<<")
    else:
        print("\n没有可比对的结果 (请先运行 HLS C 仿真并设置 HLS_PGM_DIR)")
