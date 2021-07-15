#pragma once
#include <vector>
#include <memory>
#include <stack>
#include <Dense>
#include "ActivationFunction.hpp"

void show_loading_bar(int iteration, const int& number_of_iterations, const int& bar_size, double& next_report, const double& report_interval);

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
    NeuralNetwork(int num_layers, const std::vector<int>& num_neurons) {
        this->activation_function = std::make_shared<TActivationFunction>();
        assert(num_neurons.size() == num_layers);
        this->num_layers = num_layers;
        this->num_neurons = num_neurons;
        this->weights.reserve(num_layers - 1);
        this->biases.reserve(num_layers - 1);
        for (int i = 0; i < num_layers - 1; i++)
        {
            this->weights.push_back(Eigen::MatrixXd::Random(num_neurons[i + 1], num_neurons[i]) / 1000);
            this->biases.push_back(Eigen::MatrixXd::Ones(num_neurons[i + 1], 1));
        }
    }

    NeuralNetwork(const std::string& filename) {
        this->activation_function = std::make_shared<TActivationFunction>();
        std::ifstream file(filename);
        if (file.is_open())
        {
            file >> this->num_layers;
            this->num_neurons = std::vector<int>(this->num_layers);
            file >> this->num_neurons.front();
            this->weights.reserve(num_layers - 1);
            this->biases.reserve(num_layers - 1);
            int weight_dimension[2], bias_dimension;
            for (int l = 0; l < num_layers - 1; l++)
            {
                file >> weight_dimension[0] >> weight_dimension[1];
                //this->num_neurons[0] = weight_dimension[1];
                this->weights.push_back(
                    Eigen::MatrixXd::Zero(
                        weight_dimension[0], weight_dimension[1]));
                for (int i = 0; i < weight_dimension[0]; i++)
                for (int j = 0; j < weight_dimension[1]; j++)
                {
                    double x;
                    file >> x;
                    //std::cout << x << " ";
                    this->weights[l](i, j) = x;
                }
                //std::cout << "\n";
                file >> bias_dimension;
                this->biases.push_back(
                    Eigen::MatrixXd::Ones(bias_dimension, 1));
                for (int j = 0; j < bias_dimension; j++)
                {
                    double x;
                    file >> x;
                    this->biases[l](j, 0) = x;
                }
            }
        }
    }

    void export_model(const std::string& filename) const
    {
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << num_layers << " " << num_neurons[0] << "\n";
            for (int i = 0; i < num_layers - 1; i++)
            {
                file << num_neurons[i+1]<< " " << num_neurons[i] << "\n";
                for (auto x : this->weights[i].reshaped<Eigen::RowMajor>())
                    file << x << " ";
                file << "\n";
                file << num_neurons[i+1] << "\n";
                for (auto x : this->biases[i].reshaped<Eigen::RowMajor>())
                    file << x << " ";
                file << "\n";
            }
        }
    }


    void train(const Eigen::MatrixXd& dataset, const Eigen::MatrixXd& expected_outputs, double alpha = 0.7) {
        int number_of_examples = dataset.rows();
        constexpr int number_of_iterations = TEpochs;

        auto bar = LoadingBar(40, number_of_iterations);
        
        for (int iteration = 0; iteration < number_of_iterations; iteration++, bar.show_loading_bar(iteration)){
            ;
            for (int example = 0; example < number_of_examples; example++)
            {
                auto deltas = train_example(
                    dataset.row(example).transpose(),
                    expected_outputs.row(example).transpose()
                );

                // updating weights
                for (int layer = this->num_layers - 2; layer >= 0; layer--) {
                    // Weights delta
                    this->weights[layer] += alpha * deltas.first[layer];
                    // Biases delta
                    this->biases[layer] += alpha * deltas.second[layer];
                }

            }
        }
    }

    std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> 
    train_example(
        const Eigen::MatrixXd& input,
        const Eigen::MatrixXd& output,
        double alpha = 0.7)
    {
        std::vector<Eigen::MatrixXd> weight_deltas(this->num_layers - 1);
        std::vector<Eigen::MatrixXd> bias_deltas(this->num_layers - 1);

        // Feed forward the current input value
        std::vector<Eigen::MatrixXd> z_values;
        std::vector<Eigen::MatrixXd> activations;
        Eigen::MatrixXd cost = output - this->feed_forward(z_values, activations, input);

        // Since there's no z-value in the first (input) layer
        Eigen::MatrixXd delta =
            cost.cwiseProduct(this->activation_function->apply_derivative(z_values.back()));

        weight_deltas[this->num_layers - 2] = delta * activations[this->num_layers - 2].transpose();
        bias_deltas[this->num_layers - 2] = delta;

        for (int layer = this->num_layers - 3; layer >= 0; layer--)
        {
            delta = (this->weights[layer + 1].transpose() * delta).cwiseProduct(
                this->activation_function->apply_derivative(z_values[layer])
            );
            weight_deltas[layer] = delta * activations[layer].transpose();
            bias_deltas[layer] = delta;
        }

        return std::make_pair(weight_deltas, bias_deltas);
    }

    /** Simply pass a list of inputs through the network and return
    *   the predicted results
    */
    Eigen::MatrixXd feed_forward(const Eigen::MatrixXd& input) const
    {
        assert(input.rows() == num_neurons[0]);
        auto layer_result = input;
        // Feed forward storing the results for each layer
        for (int layer_id = 0; layer_id < this->num_layers - 1; layer_id++) {
            // Applying the layer weights and the activation function
            layer_result = activation_function->apply_function(
                weights[layer_id] * layer_result + this->biases[layer_id]);
        }
        return layer_result;
    }


    /** Another version of the feed forward step used in the training. This
    *   version is different in that it stores the intermediate Z values and
    *   each of its activation results for each layer.
    *
    *   Obs.: The neuron values of an intermediate layer can be defined as a_{l},
    *   where a_{l} = \sigma{ z_{l} }, and z_{l} = W{l} * a_{l-1} + b{l}.
    *   There z_{l} values are used during the backpropagation step in training, and
    *   thus, need to be stored.
    */
    Eigen::MatrixXd feed_forward(
        std::vector<Eigen::MatrixXd>& zs,
        std::vector<Eigen::MatrixXd>& activations,
        const Eigen::MatrixXd& input) const
    {
        assert(input.rows() == num_neurons[0]);
        activations.push_back(input);
        Eigen::MatrixXd layer_result = activations.back();
        // Feed forward storing the results for each layer
        for (int layer_id = 0; layer_id < this->num_layers - 1; layer_id++) {
            // Applying the layer weights and the activation function
            zs.push_back(this->weights[layer_id]*layer_result + this->biases[layer_id]);
            layer_result = activation_function->apply_function(zs.back());
            activations.push_back(layer_result);
        }
        return layer_result;
    }
};


class LoadingBar {
public:
    int bar_size;
    int max_progress;
    double report_interval;
    double next_report;

    LoadingBar(
        int bar_size,
        double max_progress=100,
        char line_separator = '\n'
    ) :   
        bar_size(bar_size), 
        next_report(0.0), 
        report_interval(100.0/bar_size), 
        max_progress(max_progress) {}

    void show_loading_bar(double progress){
        double progress_fraction = progress / this->max_progress;
        double bar_ticks = this->bar_size * progress_fraction;
        double percentage = 100 * progress_fraction;
        if (percentage >= this->next_report) {
            std::cout << "Training progress |"
                << std::string(bar_ticks, '#')
                << std::string(this->bar_size - bar_ticks, '.')
                << "| " << percentage << "%\n";
            this->next_report += this->report_interval;
        }
    }
};
