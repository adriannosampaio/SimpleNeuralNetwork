# SimpleNeuralNet Library core

add_library(SimpleNeuralNet src/NeuralNet.cpp)

target_include_directories(SimpleNeuralNet PUBLIC include)
target_link_libraries(SimpleNeuralNet PUBLIC eigen)