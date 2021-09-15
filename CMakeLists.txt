# CMakeList.txt: projeto de CMake para CMakeProject1, incluir origem e definir
# lógica específica a um projeto aqui.
#
cmake_minimum_required (VERSION 3.8)

project ("CMakeProject1")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# Catch2 library
find_package(Catch2 REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

find_package(Eigen3 REQUIRED)
include_directories(Eigen3::Eigen INTERFACE $ENV{CONDA_PREFIX}/Library/include/eigen3)

add_subdirectory(libs/Mnist2Eigen)
add_subdirectory(libs/SimpleNeuralNet)
add_subdirectory(libs/pySimpleNeuralNet)
# Unit tests
add_subdirectory(tests)
# Adicione a origem ao executável deste projeto.
add_executable (CMakeProject1 "main.cpp")

#target_include_directories(CMakeProject1 PUBLIC libs/SimpleNeuralNet libs/Mnist2Eigen)
target_link_libraries(CMakeProject1 PRIVATE Eigen3::Eigen mnist2eigen SimpleNeuralNet)