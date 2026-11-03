# ============================================================
# run_morphology.tcl — 形态学算子: C仿真 -> C综合 -> 导出IP
# 用法(在 hls_project 目录下):  vitis_hls -f tcl/run_morphology.tcl
# ============================================================
open_project -reset proj_morphology
set_top morphology

add_files src/morphology.cpp
add_files -tb tb/tb_morphology.cpp
add_files -tb tb/pgm_io.h

open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

# ---- C 仿真 ----
csim_design -clean

# ---- C 综合 ----
csynth_design

# ---- 导出 Vivado IP (确认无误后取消注释) ----
# export_design -format ip_catalog

exit
