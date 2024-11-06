#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <chrono>

#include "model.h"
#include "testModel.h"

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
    //read data
    Eigen::MatrixXf W1 = load_csv("mydata/Weights_conv1.csv");
    Eigen::MatrixXf W2 = load_csv("mydata/Weights_conv2.csv");
    Eigen::MatrixXf adjMat = load_csv("mydata/adjMat.csv");
    Eigen::MatrixXf X = load_csv("mydata/xMat.csv");
    Eigen::MatrixXf yMat = load_csv("mydata/yMat.csv").transpose();

    Eigen::RowVectorXi Y = yMat.row(0).cast<int>();
    std::cout << "y: \n" << Y << std::endl;

    //initialize model, update weights
    model myModel = model();
    myModel.updateWeights(W1,W2);

    //Perform one pass of model, record time taken
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXf out = myModel.forward(X, adjMat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    std::cout << "Time taken for one pass through model: " << duration.count() << " microseconds" << std::endl;

    Eigen::RowVectorXi pred = argmax(out);

    std::cout << "pred:\n" << pred << std::endl;

    float acc = accuracy(Y, pred);

    std::cout << "Accuracy: " << acc << std::endl;

    float mse = MSE(Y,pred);

    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;

    return 0;
}