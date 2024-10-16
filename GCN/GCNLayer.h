#ifndef GCNLAYER_H_
#define GCNLAYER_H_

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>

class GCNLayer
{
    public:
        GCNLayer(int input_dim, int output_dim, Eigen::MatrixXd adjMatrix);
        Eigen::MatrixXd forward(Eigen::MatrixXd X);
    private:
        Eigen::MatrixXd A;
        Eigen::MatrixXd A_hat;
        Eigen::MatrixXd D;
        Eigen::MatrixXd W;
};

#endif
