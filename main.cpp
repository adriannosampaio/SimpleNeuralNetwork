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
	/*
	Eigen::MatrixXd weights = Eigen::MatrixXd::Random(3, 1);
	std::cout << dataset << "\n" << output << "\n" << weights << "\n";

	auto function = Sigmoid();
	
	// Training
	for (int iteration = 0; iteration < 10000; iteration++)
	{
		// First multiply the dataset by the weights to test the results
		Eigen::MatrixXd res = dataset * weights;
		res = function.apply_function(res);
		if(iteration % 2000 == 0)
			std::cout << res << "\n\n";
		
		// Check distance from expected values
		Eigen::MatrixXd error = output - res;

		// Use the error as factor multiplied by the activation
		// function derivative to indicate the "step in the right 
		// direction"
		Eigen::MatrixXd delta(error);
		auto derivatives = function.apply_derivative(res);
		for (int i = 0; i < delta.rows(); i++){
			for (int j = 0; j < delta.cols(); j++){
				delta(i, j) *= derivatives(i, j);
			}
		}
		
		Eigen::MatrixXd step = dataset.transpose() * delta;
		weights += step;
	}
	*/

}
