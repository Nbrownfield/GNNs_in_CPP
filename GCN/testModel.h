#include <cmath>
#include <eigen3/Eigen/Dense>

//Accuracy = # of correct predictions / total # of predictions
float accuracy(Eigen::RowVectorXi Y, Eigen::RowVectorXi pred_Y);

//MSE = 1/N * sum((actual-predicted)^2)
float MSE(Eigen::RowVectorXi Y, Eigen::RowVectorXi pred_Y);

//returns vector of col index of max value in each row
Eigen::RowVectorXi argmax(Eigen::MatrixXf mat);