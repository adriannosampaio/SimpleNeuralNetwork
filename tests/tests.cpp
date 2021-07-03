#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <Dense>
#include "NeuralNet.hpp"
#include "MnistReader.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>

template<class TA, int TI>
void test_model(
	const NeuralNetwork<TA, TI>& model, 
	const Eigen::MatrixXd& test_set_inputs, 
	const Eigen::MatrixXd& test_set_outputs)
{
	Eigen::MatrixXd obtained_output(
		test_set_outputs.cols(),
		test_set_outputs.rows());
	
	int test_examples = test_set_inputs.rows();

	for (int ex = 0; ex < test_examples ; ex++){
		obtained_output.col(ex) = model.feed_forward(test_set_inputs.row(ex).transpose());
	}

	int num_errors = 0;
	for (int ex = 0; ex < test_examples; ex++){
		for (int i = 0; i < test_set_outputs.cols(); i++)
		{
			auto result_difference = obtained_output(i, ex) - test_set_outputs(ex, i);
			bool is_correct = result_difference == Catch::Detail::Approx(0.0).margin(0.1);
			CHECK(is_correct);
			if (!is_correct)
			{
				WARN("Error at example " << ex << " , in output neuron " << i << ": " << result_difference);
			}
			num_errors += !is_correct;
		}
	}

	WARN(
		num_errors << "out of " << test_examples << "cases failed (" 
		<< 100 * (double) num_errors / test_examples<< "% fail rate)"
	);
}

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
	auto nn = NeuralNetwork<Sigmoid, 1000>(3, { 3, 3, 1 });
	nn.train(dataset, expected_output);
	test_model(nn, dataset, expected_output);

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 <<
		0, 0, 0,
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;

	Eigen::MatrixXd expected_test_data(4, 1);
	expected_test_data << 1, 1, 0, 0;
	test_model(nn, dataset2, expected_test_data);

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
	auto nn = NeuralNetwork<Sigmoid, 1000>(4, {3, 3, 3, 1 });
	nn.train(dataset, expected_output);
	test_model(nn, dataset, expected_output);

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 <<
		0, 0, 0,
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;

	Eigen::MatrixXd expected_test_data(4, 1);
	expected_test_data << 0, 1, 0, 1;
	test_model(nn, dataset2, expected_test_data);
}
/*
TEST_CASE("MNIST_READ", "[mnist]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
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

	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto data = mnist2eigen::read_mnist_dataset(test_dir + "/MNIST-dataset");
	
	UNSCOPED_INFO("Start testing read results");
	for (int i = 0; i < 28; i++)
	for (int j = 0; j < 28; j++){
		int index = i * 28 + j;
		if (data.test_images(0, index) > 0.5){
			REQUIRE(non_zero_pixels.count(index) > 0);
		}
	}
}
*/
