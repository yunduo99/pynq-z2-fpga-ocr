# -*- coding: utf-8 -*-
"""
摄像头选择器 —— 分辨内置摄像头和 USB 摄像头
============================================================
作用:
  1) 同时打开电脑上所有可用摄像头, 每个开一个窗口并标上编号
  2) 你看哪个窗口是 USB 摄像头, 按对应数字键选中
  3) 选中的编号会写入 cam.txt, 之后 test4/test6 等脚本会自动使用它

用法: 双击 test0_camera_check.bat
按键: 数字键 0~3 = 选中该编号摄像头(写入 cam.txt)
      s = 截图    q = 退出
"""
import os, sys, time
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')
import cv2
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MAXIDX = 4
HERE = os.path.dirname(os.path.abspath(__file__))
CAMFILE = os.path.join(HERE, 'cam.txt')


def open_cam(idx, tries=8):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == 'nt' else [cv2.CAP_ANY]
    for be in backends:
        try:
            cap = cv2.VideoCapture(idx, be)
        except Exception:
            continue
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            continue
        for _ in range(tries):
            try:
                ok, f = cap.read()
            except Exception:
                ok, f = False, None
            if ok and f is not None:
                return cap
            time.sleep(0.15)
        try:
            cap.release()
        except Exception:
            pass
    return None


def put_cn(img, text, pos, color=(0, 255, 0), size=26):
    try:
        from PIL import Image, ImageDraw, ImageFont
        for fp in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf'):
            if os.path.isfile(fp):
                f = ImageFont.truetype(fp, size)
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ImageDraw.Draw(pil).text(pos, text, font=f,
                                         fill=(color[2], color[1], color[0]))
                return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


def main():
    print('=' * 64)
    print('  摄像头选择器（分辨内置 / USB 摄像头）')
    print('=' * 64)
    caps = {}
    print('  正在查找摄像头 ...')
    for i in range(MAXIDX):
        c = open_cam(i)
        if c is not None:
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            caps[i] = c
            print(f'    编号 {i}: ✅ 可用   {w}x{h}')
        else:
            print(f'    编号 {i}: ✗ 不可用')

    if not caps:
        print('\n  ❌ 没有找到任何摄像头')
        print('     ① 换 USB 口（台式机插机箱后面）')
        print('     ② 关掉占用摄像头的程序（相机App/微信/QQ/钉钉）')
        print('     ③ 之前运行的本程序窗口是否还开着？')
        try:
            input('\n  按回车键关闭 ...')
        except EOFError:
            pass
        return

    # 读当前已选
    cur = ''
    if os.path.isfile(CAMFILE):
        cur = open(CAMFILE, encoding='utf-8').read().strip()
    print(f'\n  当前选择: cam.txt = {cur or "(未设置, 默认 0)"}')
    print('\n  ★ 每个摄像头会弹出一个窗口, 标题写着编号')
    print('  ★ 看哪个窗口是 USB 摄像头, 就按那个数字键')
    print('  按键: 数字键=选中   s=截图   q=退出')
    print('-' * 64)

    titles = {i: f'Camera {i}  (press {i} to select)' for i in caps}
    for i in caps:
        cv2.namedWindow(titles[i], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(titles[i], 640, 480)

    t0 = time.time()
    n = 0
    while True:
        alive = []
        for i, cap in list(caps.items()):
            ok, frame = cap.read()
            if not ok:
                continue
            alive.append(i)
            n += 1
            disp = frame.copy()
            # 缩放显示
            h, w = disp.shape[:2]
            s = min(640 / w, 480 / h, 1.0)
            if s < 1:
                disp = cv2.resize(disp, (int(w * s), int(h * s)))
            label = f'编号 {i}'
            if cur and str(i) == cur:
                label += '  ← 当前选用'
            disp = put_cn(disp, label, (12, 12), (0, 255, 0), 30)
            fps = n / max(time.time() - t0, 0.001)
            cv2.putText(disp, f'{w}x{h}  {fps:.0f} FPS  press {i} to select',
                        (12, disp.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 1)
            cv2.imshow(titles[i], disp)
        if not alive:
            print('  ⚠️ 所有摄像头都读不到画面了')
            break
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('s'):
            for i, cap in caps.items():
                ok, f = cap.read()
                if ok:
                    fn = f'shot_cam{i}.jpg'
                    cv2.imwrite(fn, f)
                    print(f'  → 已保存 {fn}')
        ch = chr(k) if k < 128 else ''
        if ch.isdigit() and int(ch) in caps:
            cur = ch
            with open(CAMFILE, 'w', encoding='utf-8') as f:
                f.write(ch)
            print(f'  ✅ 已选用 编号 {ch}（已写入 cam.txt）')
            print(f'     之后 test4/test6 等脚本会自动使用这个摄像头 ✓')

    for cap in caps.values():
        cap.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print('-' * 64)
    print(f'  最终选择: 编号 {cur or "(未改)"}   文件 {CAMFILE}')


if __name__ == '__main__':
    main()
