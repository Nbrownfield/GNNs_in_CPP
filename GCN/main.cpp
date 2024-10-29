#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

#include "model.h"

Eigen::MatrixXf load_csv(const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<float> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
}

double cross_entropy(Eigen::MatrixXf y_true, Eigen::MatrixXf y_pred)
{
    double loss = 0.0;
    for (int i = 0; i < y_true.rows(); i++)
    {
        for (int j = 0; j < y_true.cols(); j++)
        {
            loss += y_true(i, j) * log(y_pred(i, j));
        }
    }
    return -loss;
}

int main()
{
    Eigen::MatrixXf W1 = load_csv("mydata/Weights_conv1.csv");
    Eigen::MatrixXf W2 = load_csv("mydata/Weights_conv2.csv");
    Eigen::MatrixXf adjMat = load_csv("mydata/adjMat.csv");
    Eigen::MatrixXf X = load_csv("mydata/xMat.csv");

    model myModel = model(adjMat);
    myModel.updateWeights(W1,W2);
    Eigen::MatrixXf out = myModel.forward(X);

    std::cout << "out:\n" << out << std::endl;

    return 0;
}