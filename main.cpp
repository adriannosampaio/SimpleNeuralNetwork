// CMakeProject1.cpp: define o ponto de entrada para o aplicativo.
//
#include <iostream>
#include <vector>
#include <set>
#include <Dense>

#include "MnistReader.hpp"
#include "NeuralNet.hpp"

int main()
{
	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	mnist2eigen::MNISTData&& data = 
		mnist2eigen::read_mnist_dataset("../tests/MNIST-dataset/");
	mnist2eigen::write_ppm("test.ppm", data.test_images, 10);

	auto nn = NeuralNetwork({ 28 * 28, 100, 10 }, {"relu", "relu"}, "quadratic");

	// Converting labels to one-hot encoded data
	Eigen::MatrixXd expected_outputs(data.train_labels.rows(), 10);
	for (int r = 0; r < data.train_images.rows(); r++){
		for (int i = 0; i < 10; i++){
			expected_outputs(r, i) =
				(i == data.train_images(r)) ? 1.0 : 0.0;
		}
	}
	nn.train(
		data.train_images.block(0, 0, data.train_images.cols(), 200),
		expected_outputs.block(0, 0, expected_outputs.cols(), 200));

	return 0;
}
