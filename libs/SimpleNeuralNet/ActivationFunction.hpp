#pragma once
#include <memory>

class ActivationFunction
{
public:
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
    inline double function(double x) const override {
        return 1 / (1 + exp(-x));
    }

    inline double derivative(double x) const override {
        return this->function(x) * (1 - this->function(x));
    }
};

std::shared_ptr<ActivationFunction> activation_from_name(const std::string& function_name)
{
    if(function_name == "sigmoid") return std::make_shared<Sigmoid>();
    // If the function did not yet return a value, then the passed function
    // name does not have a class defined
    throw std::runtime_error("Invalid activation function name: " + function_name);
}