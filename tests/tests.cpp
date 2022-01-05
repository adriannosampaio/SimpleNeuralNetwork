#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>
#include <Eigen/Dense>
#include "NeuralNet.hpp"
#include "MnistReader.hpp"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <memory>

void test_mnist_model(
	const NeuralNetwork& model,
	const Eigen::MatrixXd& test_set_inputs,
	const Eigen::MatrixXd& test_set_outputs)
{
	//Eigen::MatrixXd obtained_output(
	//	test_set_outputs.cols(),
	//	test_set_outputs.rows());

	int test_examples = test_set_inputs.rows();
	int wrong_predictions = 0;
	for (int ex = 0; ex < test_examples; ex++) {
		int row, col;
		// Getting the row of the maximum value in the expected output
		test_set_outputs.row(ex).transpose().maxCoeff(&row, &col);
		// this will get the actual digit expected
		int digit_expected = row;
		// Obtaining a prediction for the test case
		Eigen::MatrixXd prediction = model.feed_forward(test_set_inputs.row(ex).transpose());
		// finding the maximum value in the output array (thus, finding the predicted digit)
		prediction.maxCoeff(&row, &col);
		// get the predicted digit
		int digit_predicted = row;
		if (digit_expected != digit_predicted)
			wrong_predictions++;
	}

	CHECK(wrong_predictions < test_examples);
	WARN(wrong_predictions << " out of " << test_examples << " failed predictions ("
		<< 100 * (double)wrong_predictions / test_examples << "% fail rate)"
	);
}

void test_model(
	const NeuralNetwork& model, 
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

	auto bar = LoadingBar(40, test_examples);
	int num_errors = 0;
	for (int ex = 0; ex < test_examples; ex++, bar.show_loading_bar(ex)){
		for (int i = 0; i < test_set_outputs.cols(); i++)
		{
			auto result_difference = obtained_output(i, ex) - test_set_outputs(ex, i);
			bool is_correct = result_difference == Catch::Detail::Approx(0.0).margin(0.1);
			if (!is_correct){
				num_errors++;
				CHECK(is_correct);
				WARN(
					"Error at example " << ex 
					<< " , in output neuron " << i 
					<< ": " << result_difference << "\n"
				);
			}
		}
	}
	int total_assertions = test_examples * test_set_outputs.cols();
	CHECK(num_errors < total_assertions);
	WARN(
		num_errors << " out of " << test_examples * test_set_outputs.cols() << " checks failed ("
		<< 100 * (double) num_errors / (test_examples * test_set_outputs.cols())<< "% fail rate)"
	);
}

TEST_CASE("Invert first input (Sigmoid)", "[logic_gate][model_train][model_test]") {

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
	auto nn = NeuralNetwork(
		{3, 3, 1 }, // Number of neurons per layer
		{"sigmoid", "sigmoid"}, // Activation functions
		"quadratic"
	);
	nn.train(dataset, expected_output);
	test_model(nn, dataset, expected_output);

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 <<
		0, 0, 0,
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;

	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	nn.export_model(test_dir + "/model_sigmoid.json");
	NeuralNetwork model(test_dir + "/model_sigmoid.json");
	test_model(model, dataset, expected_output);

	Eigen::MatrixXd expected_test_data(4, 1);
	expected_test_data << 1, 1, 0, 0;
	test_model(nn, dataset2, expected_test_data);

}

TEST_CASE("(first OR second) AND third inputs", "[logic_gate][model_train][model_test]") {

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
	auto nn = NeuralNetwork(
		{3, 3, 3, 1 },
		{"sigmoid", "sigmoid", "sigmoid"},
		"quadratic"
	);
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

TEST_CASE("MNIST_READ", "[mnist][data_read]") {
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

TEST_CASE("MNIST_TRAIN_Sig_Quad", "[mnist][model_train][model_test][.]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto data = mnist2eigen::read_mnist_dataset(test_dir + "/MNIST-dataset");
	//mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	auto nn = NeuralNetwork({ 28 * 28, 30, 10 }, {"sigmoid", "sigmoid"}, "quadratic");

	// Converting labels to one-hot encoded data
	Eigen::MatrixXd expected_outputs(data.train_labels.rows(), 10);
	for (int r = 0; r < data.train_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			expected_outputs(r, i) =
				(i == data.train_labels(r)) ? 1.0 : 0.0;
		}
	}
	nn.train(data.train_images, expected_outputs, 100, 0.1);
	//data.train_images.block(0, 0, 100, data.train_images.cols()),
	//expected_outputs.block(0, 0, 100, expected_outputs.cols()));

// Converting test labels to one-hot encoded data
	Eigen::MatrixXd test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}
	test_mnist_model(nn, data.test_images, test_outputs);

	nn.export_model(test_dir + "/mnist_model_sigmoid_quadratic.json");
}

TEST_CASE("MNIST_IMPORT_Sig_Quad", "[mnist][model_test][.]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto folder = test_dir + "/MNIST-dataset";
	// Reading only test data
	const static std::string test_images_file = "t10k-images-idx3-ubyte";
	const static std::string test_labels_file = "t10k-labels-idx1-ubyte";
	mnist2eigen::MNISTImageReader test_images(folder + "/" + test_images_file);
	mnist2eigen::MNISTLabelReader test_labels(folder + "/" + test_labels_file);
	mnist2eigen::MNISTData data;
	data.test_images = test_images.get_data();
	data.test_labels = test_labels.get_data();
	//mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	// Converting test labels to one-hot encoded data
	Eigen::MatrixXd test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}

	std::ofstream file(std::string(std::getenv("SNN_DIR")) + "/tools/mnist_test_data.txt");
	// for each test case (image to be predicted)
	for (int img = 0; img < data.test_labels.rows(); img++)
	{
		// save pixels
		for (int pixel = 0; pixel < data.test_images.cols(); pixel++)
		{
			file << data.test_images(img, pixel) << " ";
		}
		file << "\n";
		// save label (with one-hot encoding)
		for (int category = 0; category < 10; category++)
		{
			file << test_outputs(img, category) << " ";
		}
		file << "\n\n";
	}



	auto nn = NeuralNetwork(test_dir + "/mnist_model_sigmoid_quadratic.json");
	test_mnist_model(nn, data.test_images, test_outputs);
}

TEST_CASE("MNIST_TRAIN", "[mnist][model_train][model_test][.]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto data = mnist2eigen::read_mnist_dataset(test_dir + "/MNIST-dataset");
	//mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	// Converting labels to one-hot encoded data
	Eigen::MatrixXd expected_training_outputs(data.train_labels.rows(), 10);
	for (int r = 0; r < data.train_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			expected_training_outputs(r, i) =
				(i == data.train_labels(r)) ? 1.0 : 0.0;
		}
	}

	// Converting test labels to one-hot encoded data
	Eigen::MatrixXd expected_test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			expected_test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}
	std::map<std::string, std::shared_ptr<NeuralNetwork>> networks;
	networks["sigmoid_quadratic"]			= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "sigmoid", "sigmoid" }, "quadratic"	 ));
	networks["relu_quadratic"]				= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "relu", "relu" },		 "quadratic"	 ));
	networks["sigmoid_crossEntropy"]		= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "sigmoid", "sigmoid" }, "cross-entropy"));
	networks["sigmoid_crossEntropy"]		= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "sigmoid", "softmax" }, "cross-entropy"));
	networks["relu_crossEntropy"]			= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "relu", "relu" },		 "cross-entropy" ));
	networks["relu_softmax_crossEntropy"]	= std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "relu", "softmax" },	 "cross-entropy" ));

	for (auto& net : networks){
		auto model_name = net.first;
		auto model = net.second;
		std::cout << "\n\n\tTraining model: " << model_name << "\n\n\n";
		model->train(data.train_images, expected_training_outputs, 100, 0.1);
		std::cout << "\n\tTraining dataset:\n\n";
		test_mnist_model(*model, data.train_images, expected_training_outputs);
		std::cout << "\n\tTest dataset:\n\n";
		test_mnist_model(*model, data.test_images, expected_test_outputs);
		model->export_model(test_dir + "/mnist_model_" + model_name + ".json");
	}


	std::map<std::string, std::shared_ptr<NeuralNetwork>> networks2;
	networks2["sigmoid_quadratic"] = std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 100, 10 }, { "sigmoid", "sigmoid" }, "quadratic"));
	networks2["sigmoid_crossEntropy"] = std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 100, 10 }, { "sigmoid", "sigmoid" }, "cross-entropy"));
	networks2["sigmoid_crossEntropy"] = std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 100, 10 }, { "sigmoid", "softmax" }, "cross-entropy"));
	networks2["relu_softmax_crossEntropy"] = std::shared_ptr<NeuralNetwork>(new NeuralNetwork({ 28 * 28, 30, 10 }, { "relu", "softmax" },	 "cross-entropy"	 ));

	for (auto& net : networks2) {
		auto model_name = net.first;
		auto model = net.second;
		std::cout << "\n\n\tTraining model: " << model_name << "\n\n\n";
		model->train(data.train_images, expected_training_outputs, 100, 0.1);
		std::cout << "\n\tTraining dataset:\n\n";
		test_mnist_model(*model, data.train_images, expected_training_outputs);
		std::cout << "\n\tTest dataset:\n\n";
		test_mnist_model(*model, data.test_images, expected_test_outputs);
		model->export_model(test_dir + "/mnist_model_28*28_100_10_100it_" + model_name + ".json");
	}
}

TEST_CASE("MNIST_IMPORT_relu_cross", "[mnist][model_test]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto folder = test_dir + "/MNIST-dataset";
	// Reading only test data
	const static std::string test_images_file = "t10k-images-idx3-ubyte";
	const static std::string test_labels_file = "t10k-labels-idx1-ubyte";
	mnist2eigen::MNISTImageReader test_images(folder + "/" + test_images_file);
	mnist2eigen::MNISTLabelReader test_labels(folder + "/" + test_labels_file);
	mnist2eigen::MNISTData data;
	data.test_images = test_images.get_data();
	data.test_labels = test_labels.get_data();
	//mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	// Converting test labels to one-hot encoded data
	Eigen::MatrixXd test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}
	auto nn = NeuralNetwork(test_dir + "/mnist_model_relu_cross.json");
	test_mnist_model(nn, data.test_images, test_outputs);
}

TEST_CASE("MNIST_IMPORT_SAVED_BEFORE", "[mnist][model_test]") {
	UNSCOPED_INFO("Test starting");

	UNSCOPED_INFO("Setting expected result");
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	UNSCOPED_INFO("Getting environment variable SNN_TEST_DIR");
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	UNSCOPED_INFO("Start reading MNIST files: " + test_dir);
	auto folder = test_dir + "/MNIST-dataset";
	// Reading only test data
	const static std::string test_images_file = "t10k-images-idx3-ubyte";
	const static std::string test_labels_file = "t10k-labels-idx1-ubyte";
	mnist2eigen::MNISTImageReader test_images(folder + "/" + test_images_file);
	mnist2eigen::MNISTLabelReader test_labels(folder + "/" + test_labels_file);
	mnist2eigen::MNISTData data;
	data.test_images = test_images.get_data();
	data.test_labels = test_labels.get_data();
	//mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	// Converting test labels to one-hot encoded data
	Eigen::MatrixXd test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}
	auto nn = NeuralNetwork(test_dir + "/mnist_model_saved.json");
	test_mnist_model(nn, data.test_images, test_outputs);
}
