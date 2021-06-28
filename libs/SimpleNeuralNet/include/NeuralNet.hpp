#pragma once
#include <vector>
#include <memory>
#include <stack>
#include <Dense>
#include "ActivationFunction.hpp"

template <typename TActivationFunction, int TEpochs=100>
class NeuralNetwork {

    /** Stores the number of layers in the (including input
    *   and output)
    */
    int num_layers;

    /** Stores the number of neurons in each layer
    */
    std::vector<int> num_neurons;

    /** This structure is a vector of weight matrices where
    *   each matrix stores all weights that connect 2 layers.
    *   Consequently this vector is of size <num_layers - 1>
    *   and each matrix is of size:
    *   <neurons in layer> X <neurons in next layer>
    */
    std::vector<Eigen::MatrixXd> weights;

    /** This structure is a vector of bias vectors where
    *   each vector stores all biases associated with each neuron
    *   in every layer. This vector is of size <num_layers - 1>.
    *   Each Vector is of size <number of neurons>
    */
    std::vector<Eigen::VectorXd> biases;

    std::shared_ptr<ActivationFunction> activation_function;

public:
    NeuralNetwork(int num_layers, const std::vector<int>& num_neurons);


    void train(const Eigen::MatrixXd& dataset, const Eigen::MatrixXd& expected_outputs, double alpha = 0.7);

    /** Simply pass a list of inputs through the network and return
    *   the predicted results
    */
    Eigen::MatrixXd feed_forward(const Eigen::MatrixXd& input) const;

};