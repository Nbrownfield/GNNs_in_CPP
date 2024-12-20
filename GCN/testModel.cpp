#include "testModel.h"

//Accuracy = # of correct predictions / total # of predictions
float accuracy(Eigen::RowVectorXi Y, Eigen::RowVectorXi pred_Y)
{
    float num = 0;
    for (int i=0; i<Y.size(); i++)
    {
        if (Y[i] == pred_Y[i])
        {
            num++;
        }
    }

    return num / float(Y.size());
}

//MSE = 1/N * sum((actual-predicted)^2)
float MSE(Eigen::RowVectorXi Y, Eigen::RowVectorXi pred_Y)
{
    float num = 0;
    for (int i=0; i<Y.size(); i++)
    {
        num += std::pow((Y[i]-pred_Y[i]),2);
    }

    return num / float(Y.size());
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