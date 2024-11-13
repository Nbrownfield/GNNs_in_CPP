#include "testModel.h"

//Accuracy = # of correct predictions / total # of predictions
float accuracy(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y)
{
    float num = 0;
    for (int j=0; j<Y.cols(); j++)
    {
        for (int i=0; i<Y.rows(); i++)
        {
            if (Y(i,j) == pred_Y(i,j))
            {
                num++;
            }
        }
    }

    return num / float(Y.cols() * Y.rows());
}

//MSE = 1/N * sum((actual-predicted)^2)
float MSE(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y)
{
    float num = 0;
    for (int j=0; j<Y.cols(); j++)
    {
        for (int i=0; i<Y.rows(); i++)
        {
            num += std::pow((Y(i,j)-pred_Y(i,j)),2);
        }
    }

    return num / float(Y.cols() * Y.rows());
}

float f1Score(Eigen::MatrixXf Y, Eigen::MatrixXf pred_Y)
{
    float TP = 0;
    float FP = 0;
    float FN = 0;
    for (int j=0; j<Y.cols(); j++)
    {
        for (int i=0; i<Y.rows(); i++)
        {
            if (Y(i,j) == 1 && pred_Y(i,j) == 1)
            {
                TP++;
            }
            else if (Y(i,j) == 0 && pred_Y(i,j) == 1)
            {
                FP++;
            }
            else if (Y(i,j) == 1 && pred_Y(i,j) == 0)
            {
                FN++;
            }
        }
    }

    return TP / (TP + 0.5*(FP + FN));
}

Eigen::RowVectorXi argmax(Eigen::MatrixXf mat)
{
    Eigen::RowVectorXi out(34);
    int vecIndex = 0;
    for (int j=0; j < mat.rows(); j++)
    {
        float rowMax = 0;
        int rowMaxIndex = 0;
        for (int i=0; i < mat.cols(); i++)
        {
            if (mat(j,i) > rowMax)
            {
                rowMax = mat(j,i);
                rowMaxIndex = i;
            }
        }
        out[vecIndex] = rowMaxIndex;
        vecIndex++;
    }

    return out;
}