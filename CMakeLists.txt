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

# Eigen library
add_subdirectory(deps/Eigen)

# Catch2 library
find_package(Catch2 REQUIRED)
# Unit tests

add_subdirectory(libs/Mnist2Eigen)
add_subdirectory(libs/SimpleNeuralNet)
add_subdirectory(tests)

# Adicione a origem ao executável deste projeto.
add_executable (CMakeProject1 "main.cpp")

#target_include_directories(CMakeProject1 PUBLIC libs/SimpleNeuralNet libs/Mnist2Eigen)
target_link_libraries(CMakeProject1 PRIVATE eigen mnist2eigen SimpleNeuralNet)