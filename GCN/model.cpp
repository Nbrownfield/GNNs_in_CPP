#include "model.h"
#include "GCNLayer.h"

model::model()
    : conv1(34, 4)
    , conv2(4, 4)
{

}

Eigen::MatrixXf model::forward(Eigen::MatrixXf X, Eigen::MatrixXf adjMatrix)
{
    Eigen::MatrixXf h = model::conv1.forward(X, adjMatrix);
    h = h.array().tanh().matrix();
    Eigen::MatrixXf out = model::conv2.forward(h, adjMatrix);

    return out;
}

void model::updateWeights(Eigen::MatrixXf W1, Eigen::MatrixXf W2)
{
    model::conv1.updateWeights(W1);
    model::conv2.updateWeights(W2);
}