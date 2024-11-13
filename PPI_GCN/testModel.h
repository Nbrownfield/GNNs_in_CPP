#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Dense>
#include <cblas.h>
#include <cmath>

//Accuracy = # of correct predictions / total # of predictions
float accuracy(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y);

//MSE = 1/N * sum((actual-predicted)^2)
float MSE(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y);

float f1Score(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y);

//returns vector of col index of max value in each row
Eigen::RowVectorXi argmax(Eigen::MatrixXf mat);