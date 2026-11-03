# ============================================================
# run_binarize.tcl — 二值化算子: C仿真 -> C综合 -> 导出IP
# 用法(在 hls_project 目录下):  vitis_hls -f tcl/run_binarize.tcl
# ============================================================
open_project -reset proj_binarize
set_top binarize

add_files src/binarize.cpp
add_files -tb tb/tb_binarize.cpp
add_files -tb tb/pgm_io.h

open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# ---- C 仿真 (含与参考实现的自检) ----
csim_design -clean

# ---- C 综合 (看 II / Latency / 资源) ----
csynth_design

# ---- 导出 Vivado IP (确认无误后取消注释) ----
# export_design -format ip_catalog

exit
