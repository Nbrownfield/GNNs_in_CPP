#ifndef ADAM_H_
#define ADAM_H_

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h>

class Adam
{
    public:
        Adam(std::vector<Eigen::MatrixXd> params, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-08, double weight_decay = 0.0);
        double cross_entropy(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        void step();
    private:
        std::vector<Eigen::MatrixXd> params;
        double lr;
        double betas[2];
        double eps;
        double weight_decay;
        double m0;
        double v0;
};

#endif