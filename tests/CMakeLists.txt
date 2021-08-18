add_executable(tests tests.cpp)

target_include_directories(tests PRIVATE "../include")
target_link_libraries(tests PRIVATE mnist2eigen)
target_link_libraries(tests PRIVATE Catch2::Catch2)
target_link_libraries(tests PRIVATE eigen)
target_link_libraries(tests PRIVATE SimpleNeuralNet)
