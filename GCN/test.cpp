#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <torch/torch.h>

namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixXf_rm; //row major eigen typedef

double cross_entropy(Eigen::MatrixXf y_true, Eigen::MatrixXf y_pred)
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

MatrixXf_rm tensorToEigen(torch::Tensor T)
{
    float* data = T.data_ptr<float>();
    auto sizes = T.sizes();
    Eigen::Map<MatrixXf_rm> E(data, T.size(0), T.size(1));

    return E;
}

int main()
{
    std::cout << "***CROSS ENTROPY LOSS TEST***" << std::endl;
    Eigen::MatrixXf y_true(2, 3);
    y_true << 1, 0, 0,
              0, 1, 0;
    
    Eigen::MatrixXf y_pred(2, 3);
    y_pred << 0.7, 0.2, 0.1,
              0.25, 0.8, 0.3;

    double result = cross_entropy(y_true, y_pred);
    std::cout << "Cross Entropy Loss: " << result << std::endl;

    std::cout << std::endl << "***ITERATING THROUGH MATRIX TEST***" << std::endl;
    Eigen::MatrixXf mat(2, 3);
    mat << 1, 2, 3,
           4, 5, 6;

    for (auto row : mat.rowwise())
    {
        for (auto iter = row.begin(); iter < row.end(); iter++)
        {
            *iter = 0;
        }
    }

    std::cout << mat << std::endl;

    std::cout << std::endl << "***TENSOR TO EIGEN CONVERSION***" << std::endl;
    std::cout << "Testing libtorch to eigen: " << std::endl;

    //LibTorch
    torch::Tensor T = torch::rand({3, 3});
    std::cout << "Libtorch:\n" << T << std::endl;

    //Eigen
    MatrixXf_rm E = tensorToEigen(T);
    std::cout << "EigenMat:\n" << E << std::endl;

    E(0,0) = 0;
    std::cout << "EigenMat:\n" << E << std::endl;
    std::cout << "Libtorch:\n" << T << std::endl;


    /*std::cout << std::endl << "***DATABASE TENSOR TO EIGEN CONVERSION***" << std::endl;

    py::scoped_interpreter guard{}; //start python interpreter

    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    path.attr("append")("..");

    //import python module
    py::module myPyMod = pybind11::module::import("dataset");

    //get reference to python function
    py::function myPyFunc = myPyMod.attr("my_py_function");

    py::object pyRes = myPyFunc(T);

    T = pyRes.cast<torch::Tensor>();

    std::cout << T << std::endl;*/

    return 0;
}

