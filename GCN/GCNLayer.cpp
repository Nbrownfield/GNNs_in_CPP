#include "GCNLayer.h"

GCNLayer::GCNLayer(int input_dim, int output_dim)
{
    //std::cout << "D^(-1/2):\n";
    //std::cout << GCNLayer::D << "\n\n";

    //initalize weight matrix
    srand(40);
    GCNLayer::W = Eigen::MatrixXf::Random(input_dim, output_dim);

    //std::cout << "Weight Matrix (W):\n";
    //std::cout << GCNLayer::W << "\n\n";
}

Eigen::MatrixXf GCNLayer::forward(Eigen::MatrixXf X, Eigen::MatrixXf adjMatrix)
{
    //A_hat = A + I (adjacency matrix + identity matrix) [each node is adjacent to itself]
    Eigen::MatrixXf A_hat = adjMatrix + Eigen::MatrixXf::Identity(adjMatrix.rows(), adjMatrix.cols());

    //std::cout << "A_hat:\n";
    //std::cout << A_hat << "\n\n";

    //D = diagonal degree matrix, diagonals represent node degree (number of neighbors for each node)
    //ie element d_0 in D(0,0) represents the number of neighbors that node #0 has
    //D = A * Ones, discard elements not in diagonal
    Eigen::MatrixXf D = adjMatrix * Eigen::MatrixXf::Ones(adjMatrix.rows(), adjMatrix.cols());

    //turn diagonal into vector, then return to matrix in order discard elements not in diagonal
    Eigen::VectorXf dVector = D.diagonal();
    D = dVector.asDiagonal();

    //std::cout << "Diagonal degree matrix (D):\n";
    //std::cout << D << "\n\n";

    // D^(-1/2)
    for (int i=0;i<adjMatrix.rows();i++)
    {
        D(i,i) = std::pow(D(i,i), -0.5);
    }

    //D^(-1/2) * A_hat * D^(-1/2)
    Eigen::MatrixXf temp1 = D * A_hat * D;

    //(D^(-1/2) * A_hat * D^(-1/2)) * (X * W)
    Eigen::MatrixXf temp2 = temp1 * (X * GCNLayer::W);

    return temp2;
}

void GCNLayer::updateWeights(Eigen::MatrixXf new_W)
{
    GCNLayer::W = new_W;

    //std::cout << "new weights (W):\n";
    //std::cout << GCNLayer::W << "\n\n";
}