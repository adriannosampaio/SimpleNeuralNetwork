#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch.hpp>

#include <iostream>
#include <cstdlib>
#include <vector>
#include <Dense>
#include <NeuralNet.hpp>
#include "MnistReader.hpp"

TEST_CASE("Invert first input", "[not first gate]") {

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
	auto nn = NeuralNetwork<Sigmoid, 300>(4, { 3, 3, 3, 1 });
	nn.train(dataset, expected_output);
	auto training_data_output = nn.feed_forward(dataset);
	
	for (int row = 0; row < training_data_output.rows(); row++){
		double error = training_data_output(row, 0) - expected_output(row, 0);
		REQUIRE(error == Catch::Detail::Approx(0.0).margin(0.1));
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
	for (int row = 0; row < test_data_output.rows(); row++) {
		REQUIRE(test_data_output(row, 0) - expected_test_data(row, 0) == Catch::Detail::Approx(0.0).margin(0.1));
	}
}

TEST_CASE("(first OR second) AND third inputs", "[first or second gate]") {

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
	// (a v b) ^ c
	Eigen::MatrixXd expected_output(6, 1);
	expected_output << 0, 0, 1, 1, 0, 1;

	// Creating a simple neural network with 2 layers
	// (input and output) and 3 input nodes.
	auto nn = NeuralNetwork<Sigmoid, 300>(4, { 3, 3, 3, 1 });
	nn.train(dataset, expected_output);
	auto training_data_output = nn.feed_forward(dataset);
	
	for (int row = 0; row < training_data_output.rows(); row++){
		double error = training_data_output(row, 0) - expected_output(row, 0);
		REQUIRE(error == Catch::Detail::Approx(0.0).margin(0.1));
	}

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 <<
		0, 0, 0,
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;

	Eigen::MatrixXd expected_test_data(4, 1);
	expected_test_data << 0, 1, 0, 1;

	auto test_data_output = nn.feed_forward(dataset2);
	for (int row = 0; row < test_data_output.rows(); row++) {
		REQUIRE(test_data_output(row, 0) - expected_test_data(row, 0) == Catch::Detail::Approx(0.0).margin(0.1));
	}
}
TEST_CASE("MNIST_READ", "[mnist]") {
	INFO("Test starting");

	INFO("Setting expected result");
	std::set<int> non_zero_pixels = {
		203, 204, 205, 230, 231, 232, 233, 234, 235,
		236, 237, 238, 239, 240, 241, 242, 243, 244,
		262, 263, 264, 265, 266, 267, 268, 269, 270,
		271, 272, 273, 299, 300, 327, 328, 354, 355,
		381, 382, 383, 409, 410, 436, 437, 438, 464,
		465, 492, 493, 519, 520, 546, 547, 548, 573,
		574, 575, 601, 602, 628, 629, 655, 656, 657,
		683, 684, 685, 711, 712, 713, 739, 740
	};

	INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	INFO("Start reading MNIST files: " + test_dir);
	auto data = mnist2eigen::read_mnist_dataset(test_dir + "/MNIST-dataset");
	
	INFO("Start testing read results");
	for (int i = 0; i < 28; i++)
	for (int j = 0; j < 28; j++){
		int index = i * 28 + j;
		if (data.test_images(0, index) > 0.5){
			REQUIRE(non_zero_pixels.count(index) > 0);
		}
	}
}
