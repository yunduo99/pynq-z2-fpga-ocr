# 仅 C 仿真(快速验证), 用法: vitis_hls -f tcl/csim_binarize.tcl
open_project -reset proj_csim_bin
set_top binarize
add_files src/binarize.cpp
add_files -tb tb/tb_binarize.cpp
add_files -tb tb/pgm_io.h
open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
csim_design -clean
exit
