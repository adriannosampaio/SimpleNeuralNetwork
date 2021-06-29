#include "ActivationFunction.hpp"
#include "NeuralNet.hpp"

#include <Dense>

#include <iostream>
#include <vector>
#include <memory>
#include <stack>

template <typename TActivationFunction, int TEpochs>
NeuralNetwork<TActivationFunction, TEpochs>::NeuralNetwork(int num_layers, const std::vector<int>& num_neurons) {
    this->activation_function = std::make_shared<TActivationFunction>();
    assert(num_neurons.size() == num_layers);
    this->num_layers = num_layers;
    this->num_neurons = num_neurons;
    this->weights;
    for (int i = 0; i < num_layers - 1; i++)
    {
        this->weights.push_back(
            Eigen::MatrixXd::Random(num_neurons[i], num_neurons[i + 1]));
    }
}

template <typename TActivationFunction, int TEpochs>
void NeuralNetwork<TActivationFunction, TEpochs>::train(const Eigen::MatrixXd& dataset, const Eigen::MatrixXd& expected_outputs, double alpha = 0.7) {

    std::stack<Eigen::MatrixXd> layer_feed;
    layer_feed.push(dataset);

    constexpr int number_of_iterations = TEpochs;
    for (int iteration = 0; iteration < number_of_iterations; iteration++)
    {
        const static int bar_size = 20;
        double fraction = (double)iteration / number_of_iterations;
        int bar_ticks = bar_size * fraction;
        std::cout << "\rprogress => [" << std::string(bar_ticks, '#')
            << std::string(bar_size - bar_ticks, '.') << "]" << 100*fraction << "%" 
            << std::string(10, ' ');
            
        // Feed forward storing the results for each layer
        for (int layer_id = 0; layer_id < this->num_layers - 1; layer_id++) {
            // Applying the layer weights
            Eigen::MatrixXd res = layer_feed.top() * weights[layer_id];
            // Applying the activation function
            layer_feed.push(activation_function->apply_function(res));
        }

        // backpropagation step for each layer until input
        Eigen::MatrixXd expected_result = expected_outputs;
        // final obtained results, i.e., results of the last layer
        Eigen::MatrixXd obtained_result = layer_feed.top();
        // calculating the error of the output layer by comparing it with 
        // the expected results
        Eigen::MatrixXd layer_error = expected_result - obtained_result;

        // iterating through the layer from last to first
        for (int layer_id = this->num_layers - 1; layer_id > 0; layer_id--) {
            // getting the previous layer valus, i.e. the inputs for the current layer 
            layer_feed.pop();
            Eigen::MatrixXd layer_input = layer_feed.top();
                
            // Calculating the activation function derivatives for the obtained layer result
            Eigen::MatrixXd derivatives = this->activation_function->apply_derivative(obtained_result);
            // calculating the deltas proportionally to the 
            Eigen::MatrixXd delta = layer_error.cwiseProduct(derivatives);
            // updating the layer error for the previous layer
            layer_error = delta * this->weights[layer_id - 1].transpose();
            // calcultate the change in the weights to the current layer
            Eigen::MatrixXd step = layer_input.transpose() * delta;
            // Updating the weights
            this->weights[layer_id - 1] += alpha * step;
            // Updating previous layer obtained results by assigning the 
            // current layer input
            obtained_result = layer_input;
        }
    }
}

/** Simply pass a list of inputs through the network and return
*   the predicted results
*/
template <typename TActivationFunction, int TEpochs>
Eigen::MatrixXd NeuralNetwork<TActivationFunction, TEpochs>::feed_forward(const Eigen::MatrixXd& input) const
{
    assert(input.cols() == num_neurons[0]);
    auto layer_result = input;
    // Feed forward storing the results for each layer
    for (int layer_id = 0; layer_id < this->num_layers - 1; layer_id++) {
        // Applying the layer weights and the activation function
        layer_result = activation_function->apply_function(
            layer_result * weights[layer_id]);
    }
    return layer_result;
}