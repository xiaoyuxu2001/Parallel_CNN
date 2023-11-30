#include <random>
#include <iostream>
#include<sstream>
#include <string>
#include <vector>

#include <eigen/Eigen/Core>
#include <eigen/Eigen/Dense>
#include <Eigen/Geometry>


void printMatrix(const Eigen::MatrixXd& m)
{
  std::cout << m << std::endl;
}

//  Apply a 2D convolution operation to the input array.

Eigen::MatrixXd applyConvolution(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel)
{
    int inputRows = input.rows();
    int inputCols = input.cols();
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();

    int outputRows = inputRows - kernelRows + 1;
    int outputCols = inputCols - kernelCols + 1;

    Eigen::MatrixXd output(outputRows, outputCols);

    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            output(i, j) = (input.block(i, j, kernelRows, kernelCols).array() * kernel.array()).sum();
        }
    }

    return output;
}

// Apply a 2D max pooling operation to the input array.
Eigen::MatrixXd applyMaxPooling(const Eigen::MatrixXd& input,int pool_size = 2,int stride = 1)
{
    int inputRows = input.rows();
    int inputCols = input.cols();

    int outputRows = inputRows / pool_size;
    int outputCols = inputCols / pool_size;

    Eigen::MatrixXd output(outputRows, outputCols);

    for (int i = 0; i < outputRows; i++)
    {
        for (int j = 0; j < outputCols; j++)
        {
            double maxVal = input.block(i * pool_size, j * pool_size, pool_size, pool_size).maxCoeff();
            output(i, j) = maxVal;
        }
    }

    return output;
}

Eigen::MatrixXd flatten(Eigen::MatrixXd& input)
{
    Eigen::MatrixXd output = input.reshaped(1, input.size());

    return output;
}

class DenseLayer {
public:
    Eigen::MatrixXf weights;
    Eigen::MatrixXf biases;
    Eigen::MatrixXf z;
    Eigen::MatrixXf a;

    DenseLayer(int input_size, int output_size) {
        float limit = std::sqrt(6.0 / (input_size + output_size));
        weights = Eigen::MatrixXf::Random(input_size, output_size) * limit;
        biases = Eigen::MatrixXf::Zero(1, output_size);
    }

    Eigen::MatrixXf relu(const Eigen::MatrixXf& z) {
        return z.cwiseMax(0);
    }

    Eigen::MatrixXf softmax(const Eigen::MatrixXf& x) {
        Eigen::MatrixXf exp = x.unaryExpr([](float elem) { return std::exp(elem); });
        return exp / exp.sum();
    }

    Eigen::MatrixXf forward_pass(const Eigen::MatrixXf& x, const std::string& activation) {
        z = (x * weights).rowwise() + biases;
        if (activation == "relu") {
            a = relu(z);
        } else if (activation == "softmax") {
            a = softmax(z);
        }
        return a;
    }
};

class SequentialModel {
public:
    std::vector<std::function<Eigen::MatrixXf(Eigen::MatrixXf)>> layers;

    void add(const std::function<Eigen::MatrixXf(Eigen::MatrixXf)>& layer) {
        layers.push_back(layer);
    }

    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        Eigen::MatrixXf x = input;
        for (auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }
};

double categorical_crossentropy(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred) {
    Eigen::MatrixXf y_pred_clipped = y_pred.unaryExpr([](float elem) { return std::max(elem, 1e-7); });
    Eigen::MatrixXf cross_entropy = -(y_true.array() * y_pred_clipped.array().log()).rowwise().sum();
    return cross_entropy.mean();
}

double accuracy(const Eigen::MatrixXf& y_true, const Eigen::MatrixXf& y_pred) {
    Eigen::MatrixXf y_true_argmax = y_true.rowwise().maxCoeff();
    Eigen::MatrixXf y_pred_argmax = y_pred.rowwise().maxCoeff();
    Eigen::MatrixXf correct = (y_true_argmax - y_pred_argmax).unaryExpr([](float elem) { return std::abs(elem) < 1e-5; });
    return correct.mean();
}

class RMSpropOptimizer {
    float lr;
    float rho;
    float epsilon;
    std::vector<Eigen::MatrixXf> s;

    public:
        RMSpropOptimizer(float lr, float rho, float epsilon)
            : lr(lr), rho(rho), epsilon(epsilon) {}

        std::vector<Eigen::MatrixXf> updateWeights(const std::vector<Eigen::MatrixXf>& weights, const std::vector<Eigen::MatrixXf>& gradients) {
            if (s.empty()) {
                for (const auto& w : weights) {
                    s.push_back(Eigen::MatrixXf::Zero(w.rows(), w.cols()));
                }
            }

            std::vector<Eigen::MatrixXf> updatedWeights;
            for (size_t i = 0; i < weights.size(); ++i) {
                s[i] = rho * s[i] + (1 - rho) * gradients[i].cwiseProduct(gradients[i]);
                updatedWeights.push_back(weights[i] - lr * gradients[i].cwiseQuotient((s[i].array() + epsilon).sqrt().matrix()));
            }

            return updatedWeights;
        }
};