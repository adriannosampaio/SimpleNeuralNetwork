# SimpleNeuralNet Library core

add_library(SimpleNeuralNet src/NeuralNet.cpp include/NeuralNet.hpp include/ActivationFunction.hpp)

target_include_directories(SimpleNeuralNet PUBLIC ./include)
target_link_libraries(SimpleNeuralNet PUBLIC eigen)