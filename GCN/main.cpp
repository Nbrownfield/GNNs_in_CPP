#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

//#include "model.h"

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

int main()
{
    Eigen::MatrixXf W1 = load_csv("Weights_conv1.csv");
    std::cout << W1 << "\n\n";

    Eigen::MatrixXf W2 = load_csv("Weights_conv2.csv");
    std::cout << W2 << "\n\n";

    return 0;
}