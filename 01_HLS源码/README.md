# 阶段 1：Vitis HLS 算子 —— 已完成并通过验证 ✅

> 环境：Vitis HLS 2024.1 ｜ 器件：`xc7z020clg400-1`（PYNQ-Z2）｜ 目标时钟：100 MHz
> 验证方式：C 仿真自检 + 与板卡 OpenCV 参考图逐像素交叉比对

---

## 一、验证结果总览

| 算子 | 主循环 II | 迭代延迟 | LUT | FF | DSP | BRAM | 估算时钟(目标 10ns) | C 仿真 |
|------|----------|---------|-----|----|-----|------|-------------------|--------|
| **binarize**（二值化） | **1** ✅ | 6 周期 | 872 (1%) | 1027 | 6 (2%) | 0 | **6.912 ns** ✅ | PASS |
| **morphology**（形态学） | **1** ✅ | 5 周期 | 2244 (4%) | 1294 | 3 | 3 | **8.241 ns** ✅ | PASS |

**与 OpenCV 交叉验证（640×480 = 307200 像素）**：

| 项目 | 不一致像素 | 最大误差 | 结论 |
|------|-----------|---------|------|
| 二值化 vs `cv2.threshold(OTSU)` | **0 / 307200** | 0 | 完全一致 |
| 腐蚀 vs `cv2.erode` | **0 / 307200** | 0 | 完全一致 |
| 膨胀 vs `cv2.dilate` | **0 / 307200** | 0 | 完全一致 |
| 旁路模式（enable=0） | 0 / 307200 | 0 | 直通正确 |
| AXI-Stream sideband（TUSER/TLAST） | 0 异常 | — | 时序标记正确 |

> 参考图由**板卡上的 OpenCV 4.5.4** 生成（Otsu 阈值 = 131），保证与项目实际运行环境一致。

---

## 二、目录结构

```
hls_project/
├── src/
│   ├── binarize.cpp          二值化算子 (AXI-Stream + AXI-Lite)
│   └── morphology.cpp        3x3 形态学算子 (行缓存 + 窗口)
├── tb/
│   ├── pgm_io.h              P5 PGM 读写 (零依赖)
│   ├── tb_binarize.cpp       testbench (自检 + PGM 模式)
│   └── tb_morphology.cpp     testbench (自检 + PGM 模式)
├── tcl/
│   ├── csim_binarize.tcl     仅 C 仿真(快速)
│   ├── csim_morphology.tcl   仅 C 仿真(快速)
│   ├── run_binarize.tcl      C仿真 + C综合 (+导出IP)
│   └── run_morphology.tcl    C仿真 + C综合 (+导出IP)
├── python/
│   ├── make_testdata.py      生成测试图 + OpenCV 参考(需 cv2)
│   ├── compare_pgm.py        比对工具(需 cv2)
│   └── compare_pgm_pure.py   比对工具(零依赖, 推荐)
└── data/                     测试数据与仿真输出
    ├── input.pgm             输入灰度图 640x480
    ├── input_morph.pgm       形态学输入(= 二值图)
    ├── threshold.txt         Otsu 阈值 = 131
    ├── ref_bin/erode/dilate.pgm   OpenCV 参考结果
    └── out_bin/erode/dilate.pgm   HLS 仿真输出
```

---

## 三、怎么跑（照着做）

在 `hls_project` 目录下打开命令行：

```bat
:: 1) 仅 C 仿真（最快，验证功能）
D:\vitis2024\Vitis_HLS\2024.1\bin\vitis_hls.bat -f tcl/csim_binarize.tcl

:: 2) 带真实测试图 + 输出 PGM（用于与 OpenCV 比对）
set HLS_PGM_DIR=D:\ai.test\hls_project\data
D:\vitis2024\Vitis_HLS\2024.1\bin\vitis_hls.bat -f tcl/csim_binarize.tcl

:: 形态学（指定输入为二值图）
set HLS_PGM_IN=input_morph.pgm
D:\vitis2024\Vitis_HLS\2024.1\bin\vitis_hls.bat -f tcl/csim_morphology.tcl

:: 3) C 仿真 + C 综合（看 II 和资源，约 1 分钟）
D:\vitis2024\Vitis_HLS\2024.1\bin\vitis_hls.bat -f tcl/run_binarize.tcl
D:\vitis2024\Vitis_HLS\2024.1\bin\vitis_hls.bat -f tcl/run_morphology.tcl

:: 4) 逐像素比对（零依赖）
python python\compare_pgm_pure.py

:: 5) 导出 Vivado IP: 取消 tcl 里 export_design 一行注释后重新运行
```

> GUI 操作等价路径：双击桌面 **Vitis HLS 2024.1** → Open Project → 选 `proj_binarize`/`proj_morphology` → 左侧 `sol1` 右键 Run C Simulation / Run C Synthesis / Export RTL。

---

## 四、接口与寄存器

**统一视频接口**：24bit RGB，AXI4-Stream
- `TDATA[23:16]` = R，`[15:8]` = G，`[7:0]` = B
- `TUSER` = 帧起始（SOF），`TLAST` = 行结束（EOL）
- 控制：AXI4-Lite (`s_axilite`, bundle=control)

| 算子 | 寄存器 | 含义 |
|------|--------|------|
| binarize | `thresh` | 二值化阈值 0~255（PS 端算 Otsu 后写入） |
| | `enable` | 1=二值化，0=直通 |
| | `width` / `height` | 图像宽高（一帧像素数 = width×height） |
| morphology | `mode` | 0=直通，1=腐蚀，2=膨胀（开/闭在 BD 里串两个实例） |
| | `enable` | 使能 |
| | `width` / `height` | 图像宽高 |

---

## 五、关键设计说明（写报告可直接用）

1. **灰度系数与 OpenCV 完全一致**：采用 `(4899·R + 9617·G + 1868·B + 8192) >> 14`（ITU-R BT.601 定点，系数和 = 2^14），而非近似的 `(77,150,29)`，因此做到**逐像素零误差**。
2. **形态学边界与 OpenCV 一致**：OpenCV 默认形态学边界为常量边界，腐蚀补 255、膨胀补 0，HLS 中同样处理 → 图像四周不会出现误差。
3. **II=1 的实现关键**：
   - 3 行缓存按 `y%3` **轮转**存放（读 y-1、y-2，写 y），三者落在不同存储体，避免读写冲突；
   - `#pragma HLS ARRAY_PARTITION variable=lb dim=1 complete` 把行维展开成独立存储；
   - 列方向用 3 级移位寄存器构成窗口，避免数组随机访问。
4. **流式处理的边界刷新**：循环跑 `y=0..H`、`x=0..W`（各多一行/一列虚拟边界），从而把最后一行/列窗口补齐，输出帧尺寸仍为 H×W。
5. **资源占用极低**（LUT 1~4%、DSP 最多 6 个），为后续在 BD 里串联多级（开/闭运算）留足空间。

---

## 六、下一步

1. 取消 `tcl/run_*.tcl` 中 `export_design -format ip_catalog` 注释 → **导出 IP**；
2. Vivado 中导入 IP → 搭 Block Design（HDMI-IN → VDMA → 二值化 → 形态学 → VDMA → HDMI-OUT）；
3. 生成 `.bit` + `.hwh` → PYNQ Overlay 上板验证与性能对比。
