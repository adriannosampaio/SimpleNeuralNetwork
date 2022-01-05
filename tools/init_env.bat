@echo off
if NOT DEFINED CONDA_PREFIX (
	echo Conda not found. Please install/configure conda path.
) else (
	echo Conda exists
	echo %CONDA_PREFIX%\envs\SNN\
	setlocal EnableDelayedExpansion
	:: Define the expected path when conda is deactivate (base env)
	set SNN_CONDA_BASE_PATH=%CONDA_PREFIX%\envs\SNN\
	:: If any environment is activate, then, the base path is stored
	:: in this second variable
	if DEFINED CONDA_PREFIX_1 (
		set SNN_CONDA_BASE_PATH="%CONDA_PREFIX_1%\envs\SNN\"
	)
	if EXIST !SNN_CONDA_BASE_PATH! (
		:: If environment exists, just wait for the activation
		echo Environment found in "!SNN_CONDA_BASE_PATH!\envs\SNN". Activating..
	) else (
		:: Otherwise create environment
		echo Project environment not found. Creating...
		conda env create -f environment.yml
	)
	endlocal
	call activate SNN
	echo %~dp0\..
	set SNN_DIR=%~dp0\..
	set SNN_TEST_DIR=%~dp0\..\tests
)

