#pragma once
#include <vector>
#include <memory>
#include <stack>
#include <Dense>
#include <nlohmann/json.hpp>
#include "ActivationFunction.hpp"



class LoadingBar {
public:
    int bar_size;
    int max_progress;
    double report_interval;
    double next_report;

    LoadingBar(
        int bar_size,
        double max_progress = 100,
        char line_separator = '\n'
    ) :
        bar_size(bar_size),
        next_report(0.0),
        report_interval(100.0 / bar_size),
        max_progress(max_progress) {}

    void show_loading_bar(double progress) {
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

    std::vector<std::shared_ptr<ActivationFunction>> activation_functions;

public:
    NeuralNetwork(
        int num_layers, 
        const std::vector<int>& num_neurons,
        const std::vector<std::string>& activation_function_names) 
    {
        assert(activation_function_names.size() == num_layers - 1);
        assert(num_neurons.size() == num_layers);
        this->num_layers = num_layers;
        this->num_neurons = num_neurons;
        this->weights.reserve(num_layers - 1);
        this->biases.reserve(num_layers - 1);
        for (int i = 0; i < num_layers - 1; i++)
        {
            this->weights.push_back(Eigen::MatrixXd::Random(num_neurons[i + 1], num_neurons[i]) / 1000);
            this->biases.push_back(Eigen::MatrixXd::Ones(num_neurons[i + 1], 1));
            this->activation_functions.push_back(activation_from_name(activation_function_names[i]));
        }
    }

    NeuralNetwork(const std::string& filename) {
        using json = nlohmann::json;
        std::ifstream file(filename);
        if (file.is_open())
        {
            json model_data;
            file >> model_data;
            this->num_layers = model_data["num_layers"];
            this->num_neurons.push_back(model_data["layers"][0]["num_input_layers"]);
            for (auto j_obj : model_data["layers"])
            {
                this->num_neurons.push_back(j_obj["num_output_layers"]);
                this->activation_functions.push_back(
                    activation_from_name(j_obj["activation"].get<std::string>()));
                this->weights.push_back(
                    Eigen::MatrixXd::Zero(j_obj["num_output_layers"], j_obj["num_input_layers"]));
                for (int i = 0, idx = 0; i < j_obj["num_output_layers"]; i++)
                {
                    for (int j = 0; j < j_obj["num_input_layers"]; j++, idx++)
                    {
                        this->weights.back()(i, j) = j_obj["weights"][idx];
                    }
                }
                this->biases.push_back(
                    Eigen::MatrixXd::Ones(j_obj["num_output_layers"], 1));
                for (int j = 0; j < j_obj["num_output_layers"]; j++) {
                    this->biases.back()(j, 0) = j_obj["biases"][j];
                }
            }
        }
    }

    void export_model(const std::string& filename) const
    {
        auto indent = [](int i) { return std::string(i, '\t'); };
        std::ofstream file(filename);
        if (file.is_open())
        {
            file << "{" << "\n";
            file << indent(1) << "\"num_layers\" : " << this->num_layers << "," << "\n";
            file << indent(1) << "\"cost_function\" : " << "\"quadratic\"" << "," << "\n";
            file << indent(1) << "\"layers\" : [" << "\n";
            for (int i = 0; i < num_layers - 1; i++)
            {
                file << indent(2) << "{" << "\n";
                file << indent(3) << "\"num_input_layers\"  : " << this->weights[i].cols() << "," << "\n";
                file << indent(3) << "\"num_output_layers\" : " << this->weights[i].rows() << "," << "\n";
                file << indent(3) << "\"activation\" : " << "\""<<activation_functions[i]->get_name()<<"\"" << "," << "\n";
                file << indent(3) << "\"weights\" : " << "[";
                //file << num_neurons[i + 1] << " " << num_neurons[i] << "\n";
                int idx = 0;
                for (auto x : this->weights[i].reshaped<Eigen::RowMajor>())
                {
                    file << x;
                    if (idx != this->weights[i].size() - 1) {
                        file << ", ";
                    }
                    else {
                        file << "]," << "\n";
                    }
                    idx++;
                }
                file << indent(3) << "\"biases\" : " << "[";
                idx = 0;
                for (auto x : this->biases[i].reshaped<Eigen::RowMajor>())
                {
                    file << x;
                    if (idx != this->biases[i].size() - 1) {
                        file << ", ";
                    }
                    else {
                        file << "]" << "\n";
                    }
                    idx++;
                }
                file << indent(2) << "}" << (i == num_layers-2 ? "" : ",") << "\n";
            }
            file << indent(1) << "]" << "\n";
            file << "}";
        }
    }


    void train(const Eigen::MatrixXd& dataset, const Eigen::MatrixXd& expected_outputs, int num_iterations=1000, double alpha = 0.7) {
        int number_of_examples = dataset.rows();
        constexpr int number_of_iterations = num_iterations;

        auto bar = LoadingBar(40, static_cast<double>(number_of_iterations));
        
        for (int iteration = 0; iteration < number_of_iterations; iteration++, bar.show_loading_bar(iteration)){
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
            cost.cwiseProduct(this->activation_functions.back()->apply_derivative(z_values.back()));

        weight_deltas[this->num_layers - 2] = delta * activations[this->num_layers - 2].transpose();
        bias_deltas[this->num_layers - 2] = delta;

        for (int layer = this->num_layers - 3; layer >= 0; layer--)
        {
            delta = (this->weights[layer + 1].transpose() * delta).cwiseProduct(
                this->activation_functions[layer]->apply_derivative(z_values[layer])
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
            layer_result = activation_functions[layer_id]->apply_function(
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
            layer_result = activation_functions[layer_id]->apply_function(zs.back());
            activations.push_back(layer_result);
        }
        return layer_result;
    }
};
