#ifndef MODEL_H_
#define MODEL_H_

#include "GCNLayer.h"

#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Dense>
#include <cblas.h>
#include <iostream>
#include <cmath>
#include <math.h>

class model
{
    public:
        model();
        Eigen::MatrixXf forward(Eigen::MatrixXf X, Eigen::MatrixXf adjMatrix);
        void updateWeights(Eigen::MatrixXf W1, Eigen::MatrixXf W2);
    private:
        GCNLayer conv1;
        GCNLayer conv2;
};

#endif
