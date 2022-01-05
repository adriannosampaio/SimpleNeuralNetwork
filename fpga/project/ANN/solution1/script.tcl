############################################################
## This file is generated automatically by Vitis HLS.
## Please DO NOT edit it.
## Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
############################################################
open_project ANN
set_top feed_forward
add_files ../src/MatrixFPGA.hpp
add_files ../src/NeuralNet.cpp
add_files ../src/NeuralNet.hpp
add_files -tb ../src/testbench.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1" -flow_target vivado
set_part {xc7z020-clg400-3}
create_clock -period 10 -name default
source "./ANN/solution1/directives.tcl"
csim_design -argv {../../mnist_sig_quad_fpga.txt ../../mnist_test_data.txt}
csynth_design
cosim_design
export_design -format ip_catalog
