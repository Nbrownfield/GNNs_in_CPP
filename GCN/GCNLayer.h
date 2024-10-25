#ifndef GCNLAYER_H_
#define GCNLAYER_H_

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <math.h>

class GCNLayer
{
    public:
        GCNLayer(int input_dim, int output_dim, Eigen::MatrixXf adjMatrix);
        Eigen::MatrixXf forward(Eigen::MatrixXf X);
        void updateWeights(Eigen::MatrixXf new_W);
    private:
        Eigen::MatrixXf A;
        Eigen::MatrixXf A_hat;
        Eigen::MatrixXf D;
        Eigen::MatrixXf W; 
};

#endif
