add_library(mnist2eigen MnistReader.cpp MnistReader.hpp)
target_include_directories(mnist2eigen PUBLIC .)
target_link_libraries(mnist2eigen PUBLIC eigen)