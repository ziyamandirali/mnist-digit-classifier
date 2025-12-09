#pragma once
#include <cmath>
#include "pch.h"

float* Add_Data(float* sample, int Size, float* x, int Dim);
float* Add_Labels(float* Labels, int Size, int label);
float* init_array_random(int len);
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std);
float sgn_net(float net);

// Single-layer test (original)
int Test_Forward(float* x, float* weight, float* bias, int neuron_count, int inputDim);

// Multi-layer test (classification)
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// class_count = output layer size
int Test_Forward_MultiLayer(float* x, float** Weights, float** bias,
                             int inputDim, int* neuron_count, int Layer, int class_count);

// Multi-layer test (regression)
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// Returns continuous output value
float Test_Forward_MultiLayer_Regression(float* x, float** Weights, float** bias,
                                          int inputDim, int* neuron_count, int Layer);

float* Z_Score_Norm(float* Samples, int numSample, int inputDim);