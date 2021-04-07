# CMakeList.txt: projeto de CMake para CMakeProject1, incluir origem e definir
# lógica específica a um projeto aqui.
#
cmake_minimum_required (VERSION 3.8)

project ("CMakeProject1")


# vecsum library
add_subdirectory(vecsum)

# Eigen library
add_subdirectory(deps/Eigen)

# Adicione a origem ao executável deste projeto.
add_executable (CMakeProject1 "main.cpp")

target_include_directories(CMakeProject1 PRIVATE include)

target_link_libraries(CMakeProject1 PRIVATE vecsum)
target_link_libraries(CMakeProject1 PRIVATE eigen)


# TODO: Adicione testes e instale destinos, se necessário.
