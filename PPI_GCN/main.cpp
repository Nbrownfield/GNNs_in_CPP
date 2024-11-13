#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Dense>
#include <cblas.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
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
    Eigen::MatrixXf W1 = load_csv("data/Weights_conv1.csv");
    Eigen::MatrixXf W2 = load_csv("data/Weights_conv2.csv");
    Eigen::MatrixXf adjMat = load_csv("data/adjMat1.csv");
    Eigen::MatrixXf X = load_csv("data/xMat1.csv");
    Eigen::MatrixXf yMat = load_csv("data/yMat1.csv");
 
    //Eigen::RowVectorXi Y = yMat.row(0).cast<int>();
    std::cout << "y # of rows: " <<  yMat.rows() << std::endl << "y # of cols: " << yMat.cols() << std::endl << std::endl;

    //initialize model, update weights
    model myModel = model();
    myModel.updateWeights(W1,W2);

    //Perform one pass of model, record time taken
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXf out = myModel.forward(X, adjMat);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);

    std::cout << "Time taken for one pass through model: " << duration.count() << " microseconds" << std::endl;

    for (int j=0; j<out.cols(); j++)
    {
        for (int i=0; i<out.rows(); i++)
        {
            if (out(i,j) >= 0.5)
            {
                out(i,j) = 1;
            }
            else
            {
                out(i,j) = 0;
            }
        }
    }

    float acc = accuracy(yMat, out);
    std::cout << "Accuracy: " << acc << std::endl;

    float mse = MSE(yMat, out);
    std::cout << "MSE: " << mse << std::endl;

    float f1 = f1Score(yMat, out);
    std::cout << "F1 Score: " << f1 << std::endl;

    return 0;
}