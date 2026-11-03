# 32bit 版本: C仿真 + C综合 + 导出 IP
open_project -reset proj_mor32
set_top morphology
add_files src/morphology.cpp
add_files -tb tb/tb_morphology.cpp
add_files -tb tb/pgm_io.h
open_solution -reset sol1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default
puts "===IP=== morphology: C 仿真"
csim_design -clean
puts "===IP=== morphology: C 综合"
csynth_design
puts "===IP=== morphology: 导出 IP"
export_design -format ip_catalog
puts "===IP=== morphology 完成"
exit
