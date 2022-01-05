#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include "NeuralNet.hpp"

int main(int argc, char** argv){
	std::string tests_filename, model_filename;
	if(argc == 3)
	{
		model_filename = std::string(argv[1]);
		std::cout << model_filename << "\n";
		tests_filename = std::string(argv[2]);
		std::cout << tests_filename << "\n";
	}

	std::vector<std::vector<float>> weights(2);
	weights[0].reserve(INPUT_LAYER_SIZE*HIDDEN_LAYER_SIZE);
	weights[1].reserve(HIDDEN_LAYER_SIZE*OUTPUT_LAYER_SIZE);

	std::vector<std::vector<float>> biases(2);
	biases[0].reserve(HIDDEN_LAYER_SIZE);
	biases[1].reserve(OUTPUT_LAYER_SIZE);

	auto inputs  = std::vector<float>(); inputs.reserve(INPUT_LAYER_SIZE*MAX_NUMBER_OF_INPUTS);
	auto expected_outputs = std::vector<float>(); expected_outputs.reserve(OUTPUT_LAYER_SIZE*MAX_NUMBER_OF_INPUTS);

	auto outputs = std::vector<float>(OUTPUT_LAYER_SIZE*MAX_NUMBER_OF_INPUTS);

	bool print_examples = false;
	bool print_model = false;

	// reading inputs
	std::ifstream tests_file(tests_filename);
	if(!tests_file.is_open()){
		std::cerr << "Error opening file: " <<  tests_filename << "\n";
		exit(1);
	}
	std::string image, label;
	for(size_t example = 0; tests_file && example < MAX_NUMBER_OF_INPUTS; example++)
	{
		std::getline(tests_file, image);

		std::stringstream ss(image);
		float value = 0;
		while(ss >> value) inputs.push_back(value);


		std::getline(tests_file, label);
		std::stringstream ss_label(label);
		while(ss_label >> value) expected_outputs.push_back(value);

		if(print_examples && example < 10)
		{
			std::cout << "\n\n";
			for(int i = 0; i < INPUT_LAYER_SIZE; i++)
				std::cout << inputs[example*INPUT_LAYER_SIZE+i] << " ";
			std::cout << "\n\n";
			for(int i = 0; i < OUTPUT_LAYER_SIZE; i++)
				std::cout  << expected_outputs[example*OUTPUT_LAYER_SIZE+i] <<" ";
			std::cout << "\n\n\n";
		}
		tests_file.ignore(1, '\n');

	}


	std::ifstream model_file(model_filename);
	if(!model_file.is_open()){
		std::cerr << "Error opening file: " <<  model_filename << "\n";
		exit(1);
	}

	std::cout << "\n\nReading model...\n\n";
	std::string biases_str, weights_str;
	for(size_t layer = 0; model_file && layer < NUMBER_OF_LAYERS - 1; layer++)
	{
		std::getline(model_file, biases_str);
		std::getline(model_file, weights_str);

		float value;

		std::stringstream ssb(biases_str);
		while(ssb >> value) biases[layer].push_back(value);

		std::stringstream ssw(weights_str);
		while(ssw >> value) weights[layer].push_back(value);

		if(print_model){
			std::cout << "weight_"<<layer<<" = ";
			for(float i : weights[layer])
				std::cout << i << " ";
			std::cout << "\n\n\n\n";
			std::cout <<"biases_"<<layer<<" = ";
			for(float i : biases[layer])
				std::cout <<i << " ";
			std::cout << "\n\n\n\n";
			tests_file.ignore(1, '\n');
		}

	}

	feed_forward(
		MAX_NUMBER_OF_INPUTS,
		&inputs[0],
		&(weights[0][0]),
		&(biases[0][0]),
		&(weights[1][0]),
		&(biases[1][0]),
		&(outputs[0])
	);


	int num_errors = 0;
	for(size_t example = 0; example < MAX_NUMBER_OF_INPUTS; example++)
	{
		auto extract_digit_prediciton = [&](const std::vector<datatype>& output_vector){
			auto begin = output_vector.begin();
			auto end = output_vector.begin();
			std::advance(begin, example*OUTPUT_LAYER_SIZE);
			std::advance(end, (example + 1)*OUTPUT_LAYER_SIZE);
			auto max_element_position = std::max_element(begin, end);
			size_t expected_predicion = std::distance(begin, max_element_position);
			return expected_predicion;
		};

		size_t
			expected = extract_digit_prediciton(expected_outputs),
			obtained = extract_digit_prediciton(outputs);
		num_errors += (expected != obtained);
	}
	if(num_errors > 0){
		std::cout << "Prediction errors found: " << num_errors << std::endl;
	}

	return 0;
}
