#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch_all.hpp"

#include <iostream>
#include <vector>
#include <Dense>
#include <NeuralNet.hpp>


TEST_CASE("Invert first input", "[Not]") {

	// training input data
	Eigen::MatrixXd dataset(6, 3);
	dataset <<
		0, 0, 0,
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 0,
		1, 1, 1;

	// training expected results
	Eigen::MatrixXd expected_output(6, 1);
	expected_output << 1, 1, 1, 0, 0, 0;

	// Creating a simple neural network with 2 layers
	// (input and output) and 3 input nodes.
	auto nn = NeuralNetwork(3, { 3, 3, 1 });
	nn.train(dataset, expected_output);
	auto training_data_output = nn.feed_forward(dataset);
	
	for (int row = 0; row < training_data_output.rows(); row++){
		
		REQUIRE(training_data_output(row, 0) - expected_output(row, 0) == Catch::Approx(0).margin(.1));
	}

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 <<
		0, 0, 0,
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;


	Eigen::MatrixXd expected_test_data(4, 1);
	expected_test_data << 1, 1, 0, 0;

	auto test_data_output = nn.feed_forward(dataset2);
	for (int row = 0; row < training_data_output.rows(); row++) {
		REQUIRE(test_data_output(row, 0) == Catch::Approx(expected_test_data(row, 0)).epsilon(0.1));
	}

}