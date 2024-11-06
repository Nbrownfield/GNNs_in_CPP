Karate Club Model, pytorchImpl.py outputs necessary data for main.cpp to read into the mydata directory.

To run C++ implementation of model:

g++ -o main main.cpp model.cpp GCNLayer.cpp testModel.cpp

./main
