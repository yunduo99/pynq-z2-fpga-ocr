# -*- coding: utf-8 -*-
"""生成 HLS testbench 用的测试图 + OpenCV 参考输出

产出 (data/ 目录):
  input.pgm          输入灰度图 (HLS testbench 读取)
  threshold.txt      Otsu 阈值 (供 HLS testbench 读取, 保证与 OpenCV 同阈值)
  ref_bin.pgm        OpenCV 二值化参考结果
  ref_erode.pgm      OpenCV 腐蚀参考结果
  ref_dilate.pgm     OpenCV 膨胀参考结果
  ref_open.pgm       OpenCV 开运算参考结果
  ref_close.pgm      OpenCV 闭运算参考结果
"""
import os
import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
os.makedirs(OUT, exist_ok=True)
W, H = 640, 480

# ---------- 1. 构造测试图(含文字/形状/渐变/噪声) ----------
img = np.full((H, W), 235, np.uint8)
cv2.rectangle(img, (30, 30), (610, 150), 20, -1)
cv2.putText(img, "PYNQ Z2 2026", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 245, 4)
cv2.circle(img, (150, 300), 70, 40, -1)          # 深色圆
cv2.rectangle(img, (300, 230), (450, 380), 200, -1)  # 浅色方块
for i in range(0, W, 40):                        # 竖条纹
    cv2.line(img, (i, 400), (i, 460), 90, 3)
cv2.line(img, (0, 200), (W, 200), 120, 2)
# 轻微噪声(模拟相机)
noise = np.random.randint(0, 12, (H, W), np.uint8)
img = cv2.add(img, noise)
cv2.imwrite(os.path.join(OUT, 'input.pgm'), img)
print("已生成 input.pgm", img.shape)

# ---------- 2. Otsu 阈值 + 二值化参考 ----------
t, ref_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
with open(os.path.join(OUT, 'threshold.txt'), 'w') as f:
    f.write(f"{int(t)}\n")
cv2.imwrite(os.path.join(OUT, 'ref_bin.pgm'), ref_bin)
print(f"Otsu 阈值 = {int(t)} (已写入 threshold.txt)")

# ---------- 3. 形态学参考(3x3 矩形, OpenCV 默认边界) ----------
k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
ref_erode = cv2.erode(ref_bin, k)
ref_dilate = cv2.dilate(ref_bin, k)
ref_open = cv2.morphologyEx(ref_bin, cv2.MORPH_OPEN, k)
ref_close = cv2.morphologyEx(ref_bin, cv2.MORPH_CLOSE, k)
for name, im in [('ref_erode', ref_erode), ('ref_dilate', ref_dilate),
                 ('ref_open', ref_open), ('ref_close', ref_close)]:
    cv2.imwrite(os.path.join(OUT, f'{name}.pgm'), im)
    print("已生成", name + '.pgm')

print("\n完成。数据目录:", os.path.abspath(OUT))
print("提示: 形态学 testbench 的输入应为二值图, 若要用二值图做输入,")
print("      可把 ref_bin.pgm 复制为 input_bin.pgm 供 tb 使用。")
