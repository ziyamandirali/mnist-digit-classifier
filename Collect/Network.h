#pragma once
#include "pch.h"
#include "Process.h"

// Single-layer training (original)
float* train_fcn(float* Samples, int numSample, float* targets, int inputDim, int class_count, float* Weights, float* bias, float learning_rate, float Min_Err, int Max_epoch, int& epoch);

// Multi-layer training (classification)
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// class_count = output layer size
// momentum = 0.0 (no momentum) to 0.99 (strong momentum), typical: 0.9
float* train_fcn_multilayer(float* Samples, int numSample, float* targets, 
                            int inputDim, int* neuron_count, int Layer, int class_count,
                            float** Weights, float** bias, 
                            float learning_rate, float Min_Err, int Max_epoch, int& epoch,
                            float momentum = 0.0f);

// Multi-layer regression training
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// Output layer always has 1 neuron for regression
// momentum = 0.0 (no momentum) to 0.99 (strong momentum), typical: 0.9
float* train_fcn_multilayer_regression(float* Samples, int numSample, float* targets, 
                                       int inputDim, int* neuron_count, int Layer,
                                       float** Weights, float** bias, 
                                       float learning_rate, float Min_Err, int Max_epoch, int& epoch,
                                       float momentum = 0.0f);

// Single-layer linear regression training (original)
float* regression_train(float* x, float* y, int numSample, float& slope, float& intercept, 
                        float learning_rate, float Min_Err, int Max_epoch, int& epoch);