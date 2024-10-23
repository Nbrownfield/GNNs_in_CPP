#include "Adam.h"

//constructor initializes all arguments
Adam::Adam(std::vector<Eigen::MatrixXd> params, double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999,
            double eps = 1e-08, double weight_decay = 0.0)
{
    Adam::params = params;
    Adam::lr = lr;
    Adam::betas[0] = beta1;
    Adam::betas[1] = beta2;
    Adam::eps = eps;
    Adam::weight_decay = weight_decay;
}

double Adam::cross_entropy(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred)
{
    double loss = 0.0;
    for (int i = 0; i < y_true.rows(); i++)
    {
        for (int j = 0; j < y_true.cols(); j++)
        {
            loss += y_true(i, j) * log(y_pred(i, j));
        }
    }
    return -loss;
}

void Adam::step()
{
    //1st loop iterates through vector
    for (auto vecIter = params.begin(); vecIter < params.end(); vecIter++)
    {
        Eigen::MatrixXd mat = *vecIter;
        //2nd loop iterates through rows of selected matrix
        for (auto row : mat.rowwise())
        {
            //3rd loop iterates through elements of selected row
            for (auto elemIter = row.begin(); elemIter < row.end(); elemIter++)
            {
                //gt = derivative of loss function (cross entropy loss derivative = y_pred - y_true)
            }
        }
    }
}