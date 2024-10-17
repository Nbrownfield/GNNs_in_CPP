#include "GCNLayer.h"

GCNLayer::GCNLayer(int input_dim, int output_dim, Eigen::MatrixXd adjMatrix)
{
    GCNLayer::A = adjMatrix;

    //A_hat = A + I (adjacency matrix + identity matrix) [each node is adjacent to itself]
    GCNLayer::A_hat = adjMatrix + Eigen::MatrixXd::Identity(input_dim, input_dim);

    std::cout << "A_hat:\n";
    std::cout << GCNLayer::A_hat << "\n\n";

    //D = diagonal degree matrix, diagonals represent node degree (number of neighbors for each node)
    //ie element d_0 in D(0,0) represents the number of neighbors that node #0 has
    //D = A * Ones, discard elements not in diagonal
    GCNLayer::D = adjMatrix * Eigen::MatrixXd::Ones(input_dim, input_dim);

    //turn diagonal into vector, then return to matrix in order discard elements not in diagonal
    Eigen::VectorXd dVector = GCNLayer::D.diagonal();
    GCNLayer::D = dVector.asDiagonal();

    std::cout << "Diagonal degree matrix (D):\n";
    std::cout << GCNLayer::D << "\n\n";

    // D^(-1/2)
    for (int i=0;i<input_dim;i++)
    {
        GCNLayer::D(i,i) = std::pow(GCNLayer::D(i,i), -0.5);
    }
    std::cout << "D^(-1/2):\n";
    std::cout << GCNLayer::D << "\n\n";

    //initalize weight matrix
    srand(40);
    GCNLayer::W = Eigen::MatrixXd::Random(input_dim, output_dim);

    std::cout << "Weight Matrix (W):\n";
    std::cout << GCNLayer::W << "\n\n";
}

Eigen::MatrixXd GCNLayer::forward(Eigen::MatrixXd X)
{
    //D^(-1/2) * A_hat * D^(-1/2)
    Eigen::MatrixXd temp1 = GCNLayer::D * GCNLayer::A_hat * GCNLayer::D;

    //(D^(-1/2) * A_hat * D^(-1/2)) * (X * W)
    Eigen::MatrixXd temp2 = temp1 * (X * GCNLayer::W);

    return temp2;
}

//temp main function for testing
int main()
{
    Eigen::MatrixXd adjMatrix(3, 3);

    adjMatrix << 0, 1, 1,
                 1, 0, 0,
                 1, 0, 0;

    std::cout << "Adjacency matrix (A):\n";
    std::cout << adjMatrix << "\n\n";

    GCNLayer(adjMatrix.rows(), 4, adjMatrix);
    return 0;
}