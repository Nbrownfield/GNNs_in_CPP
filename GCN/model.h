#ifndef MODEL_H_
#define MODEL_H_

#include "GCNLayer.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include <math.h>

class model
{
    public:
        model(Eigen::MatrixXf adjMatrix);
        Eigen::MatrixXf forward(Eigen::MatrixXf X);
        void updateWeights(Eigen::MatrixXf W1, Eigen::MatrixXf W2);
    private:
        GCNLayer conv1;
        GCNLayer conv2;
};

#endif
