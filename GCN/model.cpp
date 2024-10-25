#include "model.h"
#include "GCNLayer.h"

model::model(Eigen::MatrixXf adjMatrix)
    : conv1(adjMatrix.rows(), 4, adjMatrix)
    , conv2(4, 4, adjMatrix)
{

}

Eigen::MatrixXf model::forward(Eigen::MatrixXf X)
{
    Eigen::MatrixXf h = model::conv1.forward(X);
    h = h.array().tanh().matrix();
    Eigen::MatrixXf out = model::conv2.forward(X);

    return out;
}

void model::updateWeights(Eigen::MatrixXf W1, Eigen::MatrixXf W2)
{
    model::conv1.updateWeights(W1);
    model::conv2.updateWeights(W2);
}