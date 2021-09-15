#pragma once
#include <memory>
#include <Eigen/Dense>

#include "ActivationFunction.hpp"

class LossFunction {
protected:
    std::string function_name;
public:
    LossFunction(const std::string& name) : function_name(name) {}

    /** Applies the loss function to compare the expected (y) values
    *   with the obtained (a) results of a neural network
    */
    virtual double function(const Eigen::MatrixXd& a, const Eigen::MatrixXd& y) const = 0;
    
    virtual Eigen::MatrixXd delta(
        const Eigen::MatrixXd& a, 
        const Eigen::MatrixXd& y, 
        const Eigen::MatrixXd& z, 
        std::shared_ptr<ActivationFunction> sigma) const = 0;
    
    std::string get_name() const {
        return function_name;
    }
};

class Quadratic : public LossFunction {
public:
    Quadratic() : LossFunction("quadratic") {}

    virtual double function(const Eigen::MatrixXd& a, const Eigen::MatrixXd& y) const
    {
        return 0.5 * (a - y).squaredNorm();
    }

    virtual Eigen::MatrixXd delta(
        const Eigen::MatrixXd& a, 
        const Eigen::MatrixXd& y, 
        const Eigen::MatrixXd& z, 
        std::shared_ptr<ActivationFunction> sigma) const
    {
        auto cost = a - y;
        return cost.cwiseProduct(sigma->apply_derivative(z));
    }
};

class CrossEntropy : public LossFunction
{
public:
    CrossEntropy() : LossFunction("cross-entropy") {}

    virtual double function(const Eigen::MatrixXd& a, const Eigen::MatrixXd& y) const
    {
        Eigen::MatrixXd cross_entropy = Eigen::MatrixXd::Zero(a.rows(), a.cols());
        // A and Y are always a Nx1 matrix
        for (int i = 0; i < a.rows(); i++)
        {
            // python code: np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))
            // From neuralnetworksanddeeplearning.com/chap3.html
            cross_entropy(i, 0) = (-y(i, 0) * std::log(a(i, 0))) - (1 - y(i, 0) * std::log(1 - a(i, 0)));
        }
        return cross_entropy.sum();
    }

    virtual Eigen::MatrixXd delta(
        const Eigen::MatrixXd& a, 
        const Eigen::MatrixXd& y, 
        const Eigen::MatrixXd& z, 
        std::shared_ptr<ActivationFunction> sigma) const
    {
        return (a - y);
    }
};

std::shared_ptr<LossFunction> loss_function_from_name(const std::string& function_name)
{
    if (function_name == "quadratic") return std::make_shared<Quadratic>();
    else if (function_name == "cross-entropy") return std::make_shared<CrossEntropy>();
    // If the function did not yet return a value, then the passed function
    // name does not have a class defined
    throw std::runtime_error("Invalid loss function name: " + function_name);
}