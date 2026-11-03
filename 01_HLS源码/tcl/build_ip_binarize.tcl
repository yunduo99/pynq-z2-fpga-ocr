# 32bit 版本: C仿真 + C综合 + 导出 IP
open_project -reset proj_bin32
set_top binarize
add_files src/binarize.cpp
add_files -tb tb/tb_binarize.cpp
add_files -tb tb/pgm_io.h
open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
puts "===IP=== binarize: C 仿真"
csim_design -clean
puts "===IP=== binarize: C 综合"
csynth_design
puts "===IP=== binarize: 导出 IP"
export_design -format ip_catalog
puts "===IP=== binarize 完成"
exit
