#include "GCNLayer.h"

GCNLayer::GCNLayer(int input_dim, int output_dim, Eigen::MatrixXf adjMatrix)
{
    GCNLayer::A = adjMatrix;

    //A_hat = A + I (adjacency matrix + identity matrix) [each node is adjacent to itself]
    GCNLayer::A_hat = adjMatrix + Eigen::MatrixXf::Identity(adjMatrix.rows(), adjMatrix.cols());

    std::cout << "A_hat:\n";
    std::cout << GCNLayer::A_hat << "\n\n";

    //D = diagonal degree matrix, diagonals represent node degree (number of neighbors for each node)
    //ie element d_0 in D(0,0) represents the number of neighbors that node #0 has
    //D = A * Ones, discard elements not in diagonal
    GCNLayer::D = adjMatrix * Eigen::MatrixXf::Ones(adjMatrix.rows(), adjMatrix.cols());

    //turn diagonal into vector, then return to matrix in order discard elements not in diagonal
    Eigen::VectorXf dVector = GCNLayer::D.diagonal();
    GCNLayer::D = dVector.asDiagonal();

    std::cout << "Diagonal degree matrix (D):\n";
    std::cout << GCNLayer::D << "\n\n";

    // D^(-1/2)
    for (int i=0;i<adjMatrix.rows();i++)
    {
        GCNLayer::D(i,i) = std::pow(GCNLayer::D(i,i), -0.5);
    }
    std::cout << "D^(-1/2):\n";
    std::cout << GCNLayer::D << "\n\n";

    //initalize weight matrix
    srand(40);
    GCNLayer::W = Eigen::MatrixXf::Random(input_dim, output_dim);

    std::cout << "Weight Matrix (W):\n";
    std::cout << GCNLayer::W << "\n\n";
}

Eigen::MatrixXf GCNLayer::forward(Eigen::MatrixXf X)
{
    //D^(-1/2) * A_hat * D^(-1/2)
    Eigen::MatrixXf temp1 = GCNLayer::D * GCNLayer::A_hat * GCNLayer::D;

    //(D^(-1/2) * A_hat * D^(-1/2)) * (X * W)
    Eigen::MatrixXf temp2 = temp1 * (X * GCNLayer::W);

    return temp2;
}

void GCNLayer::updateWeights(Eigen::MatrixXf new_W)
{
    GCNLayer::W = new_W;
}