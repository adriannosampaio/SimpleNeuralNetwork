@echo off
cd %SNN_DIR%
if NOT EXIST %SNN_DIR%/build ( 
	::!SNN_CONDA_BASE_PATH! (
	:: If environment exists, just wait for the activation
	echo cmake build folder not found. Creating...
	mkdir build
) 
cd build
cmake ..
start CMakeProject1.sln
cd %SNN_DIR%


