#ifndef GCNLAYER_H_
#define GCNLAYER_H_

#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Dense>
#include <cblas.h>
#include <iostream>
#include <cmath>
#include <math.h>

class GCNLayer
{
    public:
        GCNLayer(int input_dim, int output_dim);
        Eigen::MatrixXf forward(Eigen::MatrixXf X, Eigen::MatrixXf adjMatrix);
        void updateWeights(Eigen::MatrixXf new_W);
    private:
        Eigen::MatrixXf W; 
};

#endif
