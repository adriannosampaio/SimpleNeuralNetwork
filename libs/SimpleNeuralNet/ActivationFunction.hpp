#pragma once
#include <memory>

class ActivationFunction
{
protected:
    std::string function_name;
public:
    ActivationFunction(const std::string& name) : function_name(name) {}

    std::string get_name()
    {
        return this->function_name;
    }

    /** Returns the result of applying the activation function
    */
    virtual double function(double x) const = 0;

    /** Applies the activation function to every component in a
    *	matrix
    */
    virtual Eigen::MatrixXd apply_function(const Eigen::MatrixXd& matrix) const
    {
        Eigen::MatrixXd result(matrix);
        for (int i = 0; i < result.rows(); i++)
        {
            for (int j = 0; j < result.cols(); j++)
            {
                result(i, j) = this->function(result(i, j));
            }
        }
        return result;
    }

    /** Returns the result of applying the function derivative
    *	in a given point x
    */
    virtual double derivative(double x) const = 0;

    /** Applies the function derivative to every component in a
    *	matrix
    */
    virtual Eigen::MatrixXd apply_derivative(const Eigen::MatrixXd& matrix) const
    {
        Eigen::MatrixXd result(matrix);
        for (int i = 0; i < result.rows(); i++)
        {
            for (int j = 0; j < result.cols(); j++)
            {
                result(i, j) = this->derivative(result(i, j));
            }
        }
        return result;
    }
};

class Sigmoid : public ActivationFunction {
public:
    Sigmoid() : ActivationFunction("sigmoid") {}

    inline double function(double x) const override {
        return 1.0 / (1.0 + exp(-x));
    }

    inline double derivative(double x) const override {
        return this->function(x) * (1 - this->function(x));
    }
};

class ReLU : public ActivationFunction {
public:
    ReLU() : ActivationFunction("relu") {}

    inline double function(double x) const override {
        return std::max(0.0, x);
    }

    inline double derivative(double x) const override {
        // 1 if x >=0; and 0 otherwise
        return double(x > 0);
    }
};

class Softmax : public ActivationFunction {
public:
    Softmax() : ActivationFunction("softmax") {}

    inline double function(double x) const override {
        return std::max(0.0, x);
    }

    inline double derivative(double x) const override {
        // 1 if x >=0; and 0 otherwise
        return (x >= 0);
    }
};

std::shared_ptr<ActivationFunction> activation_from_name(const std::string& function_name)
{
    if (function_name == "sigmoid") return std::make_shared<Sigmoid>();
    else if (function_name == "relu") return std::make_shared<ReLU>();
    else if (function_name == "softmax") return std::make_shared<Softmax>();
    // If the function did not yet return a value, then the passed function
    // name does not have a class defined
    throw std::runtime_error("Invalid activation function name: " + function_name);
}