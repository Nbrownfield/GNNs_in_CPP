#include "GCNLayer.h"

GCNLayer::GCNLayer(int input_dim, int output_dim)
{
    //initalize weight matrix
    srand(40);
}

Eigen::MatrixXf GCNLayer::forward(Eigen::MatrixXf X, Eigen::MatrixXf adjMatrix)
{
    //A_hat = A + I (adjacency matrix + identity matrix) [each node is adjacent to itself]
    Eigen::MatrixXf A_hat = adjMatrix + Eigen::MatrixXf::Identity(adjMatrix.rows(), adjMatrix.cols());

    //D = diagonal degree matrix, diagonals represent node degree (number of neighbors for each node)
    //ie element d_0 in D(0,0) represents the number of neighbors that node #0 has
    //D = A * Ones, discard elements not in diagonal
    Eigen::MatrixXf D = adjMatrix * Eigen::MatrixXf::Ones(adjMatrix.rows(), adjMatrix.cols());

    //turn diagonal into vector, then return to matrix in order discard elements not in diagonal
    Eigen::VectorXf dVector = D.diagonal();
    for (int i = 0; i < dVector.size(); i++)
    {
        if (dVector(i) != 0)
        {
            dVector(i) = std::pow(dVector(i), -0.5);
        }
    }
    D = dVector.asDiagonal();

    //D^(-1/2) * A_hat * D^(-1/2)
    Eigen::MatrixXf temp1 = D * A_hat * D;

    //(D^(-1/2) * A_hat * D^(-1/2)) * (X * W)
    Eigen::MatrixXf xw = X * GCNLayer::W;

    Eigen::MatrixXf temp2 = temp1 * xw;

    return temp2;
}

void GCNLayer::updateWeights(Eigen::MatrixXf new_W)
{
    GCNLayer::W = new_W;
}