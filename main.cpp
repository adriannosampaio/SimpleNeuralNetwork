// CMakeProject1.cpp: define o ponto de entrada para o aplicativo.
//
#include <iostream>
#include <vector>
#include <set>
#include <Eigen/Dense>

#include "MnistReader.hpp"
#include "NeuralNet.hpp"

int main()
{

	Eigen::setNbThreads(2);
	// Pixels are already scaled from 0 to 1
	std::string test_dir = std::string(std::getenv("SNN_TEST_DIR"));
	auto folder = test_dir + "/MNIST-dataset";
	// Reading only test data
	const static std::string test_images_file = "t10k-images-idx3-ubyte";
	const static std::string test_labels_file = "t10k-labels-idx1-ubyte";
	mnist2eigen::MNISTImageReader test_images(folder + "/" + test_images_file);
	mnist2eigen::MNISTLabelReader test_labels(folder + "/" + test_labels_file);
	mnist2eigen::MNISTData data;
	data.test_images = test_images.get_data();
	data.test_labels = test_labels.get_data();

	// Converting test labels to one-hot encoded data
	Eigen::MatrixXd test_outputs(data.test_labels.rows(), 10);
	for (int r = 0; r < data.test_labels.rows(); r++) {
		for (int i = 0; i < 10; i++) {
			test_outputs(r, i) =
				(i == data.test_labels(r)) ? 1.0 : 0.0;
		}
	}

	auto filename = std::string(std::getenv("SNN_DIR")) + "/tools/mnist_test_data.txt";
	std::ofstream file(filename);
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
	file.close();
	return 0;
}
