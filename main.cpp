// CMakeProject1.cpp: define o ponto de entrada para o aplicativo.
//
#include <iostream>
#include <vector>
#include <Dense>
#include <NeuralNet.hpp>


int main()
{
	Eigen::MatrixXd dataset(6, 3);
	dataset <<  
		0, 0, 0, 
		0, 0, 1, 
		0, 1, 1, 
		1, 0, 1, 
		1, 1, 0, 
		1, 1, 1;

	Eigen::MatrixXd output(6, 1);
	output << 0, 1, 1, 1, 1, 1;

	auto nn = NeuralNetwork(3, {3, 3, 1});
	nn.train(dataset, output);
	std::cout << "test result: " << nn.feed_forward(dataset) << "\n";

	Eigen::MatrixXd dataset2(4, 3);
	dataset2 << 
		0, 0, 0, 
		0, 1, 1,
		1, 0, 0,
		1, 1, 1;
	std::cout << "test result 2: " << nn.feed_forward(dataset2) << "\n";

	return 0;
}
