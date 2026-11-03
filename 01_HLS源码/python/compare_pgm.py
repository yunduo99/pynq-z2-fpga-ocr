# -*- coding: utf-8 -*-
"""比对 HLS 仿真输出 与 OpenCV 参考结果 (逐像素)

用法:
  python compare_pgm.py                     # 比对 data/ 下全部 out_* vs ref_*
  python compare_pgm.py out_bin.pgm ref_bin.pgm
"""
import os
import sys
import cv2
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

PAIRS = [
    ('out_bin.pgm',    'ref_bin.pgm',    '二值化'),
    ('out_erode.pgm',  'ref_erode.pgm',  '腐蚀 erode'),
    ('out_dilate.pgm', 'ref_dilate.pgm', '膨胀 dilate'),
]

def compare(a_path, b_path, label):
    a = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    if a is None:
        print(f"[跳过] 找不到 HLS 输出: {a_path}")
        return None
    if b is None:
        print(f"[跳过] 找不到参考图: {b_path}")
        return None
    if a.shape != b.shape:
        print(f"[失败] 尺寸不一致: {a.shape} vs {b.shape}")
        return False
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    n_bad = int(np.count_nonzero(diff))
    total = diff.size
    print(f"\n=== {label} ===")
    print(f"  尺寸        : {a.shape[1]}x{a.shape[0]}")
    print(f"  不一致像素  : {n_bad} / {total}  ({n_bad/total*100:.4f}%)")
    print(f"  最大误差    : {int(diff.max())}")
    if n_bad == 0:
        print("  结论        : ✅ 逐像素完全一致")
        return True
    ys, xs = np.nonzero(diff)
    print(f"  首个差异点  : (x={xs[0]}, y={ys[0]})  HLS={a[ys[0], xs[0]]}  OpenCV={b[ys[0], xs[0]]}")
    print("  结论        : ❌ 存在差异")
    return False

if len(sys.argv) == 3:
    compare(os.path.join(DATA, sys.argv[1]) if not os.path.isabs(sys.argv[1]) else sys.argv[1],
            os.path.join(DATA, sys.argv[2]) if not os.path.isabs(sys.argv[2]) else sys.argv[2],
            '自定义比对')
else:
    results = []
    for a, b, label in PAIRS:
        r = compare(os.path.join(DATA, a), os.path.join(DATA, b), label)
        if r is not None:
            results.append((label, r))
    print("\n========== 汇总 ==========")
    if not results:
        print("没有可比对的结果(请先运行 HLS C 仿真生成 out_*.pgm)")
    for label, ok in results:
        print(f"  {label:14s}: {'✅ 通过' if ok else '❌ 未通过'}")
    print("  全部通过 ✅" if all(r for _, r in results) and results else "  存在未通过项 ❌")
