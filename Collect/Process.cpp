#include "pch.h"
#include "Process.h"
#include <iostream>

float* Add_Data(float* sample, int Size, float* x, int Dim) {
	float* temp = new float[Size * Dim];
	if (sample) {
		for (int i = 0; i < (Size - 1) * Dim; i++)
			temp[i] = sample[i];
		delete[] sample;
	}
	for (int i = 0; i < Dim; i++)
		temp[(Size - 1) * Dim + i] = x[i];
	return temp;
}
float* Add_Labels(float* Labels, int Size, int label) {
	float* temp = new float[Size];
	if (Labels) {
		for (int i = 0; i < Size - 1; i++)
			temp[i] = Labels[i];
		delete[] Labels;
	}
	temp[Size - 1] = float(label);
	return temp;
}
float* init_array_random(int len) {
	float* arr = new float[len];
	for (int i = 0; i < len; i++)
		arr[i] = ((float)rand() / RAND_MAX) - 0.5f;
	return arr;
}
void Z_Score_Parameters(float* x, int Size, int dim, float* mean, float* std) {

	float* Total = new float[dim];

	int i, j;
	for (i = 0; i < dim; i++) {
		mean[i] = std[i] = Total[i] = 0.0;
	}
	for (i = 0; i < Size; i++)
		for (j = 0; j < dim; j++)
			Total[j] += x[i * dim + j];
	for (i = 0; i < dim; i++)
		mean[i] = Total[i] / float(Size);

	for (i = 0; i < Size; i++)
		for (j = 0; j < dim; j++)
			std[j] += ((x[i * dim + j] - mean[j]) * (x[i * dim + j] - mean[j]));

	for (j = 0; j < dim; j++) {
		std[j] = sqrt(std[j] / float(Size));
		if (std[j] < 1e-6f) std[j] = 1.0f; // Prevent division by zero
	}

	delete[] Total;

}//Z_Score_Parameters
int Test_Forward(float* x, float* weight, float* bias, int num_Class, int inputDim) {
	int i, j, index_Max;
	if (num_Class > 2) {
		float* output = new float[num_Class];
		// Calculation of the output layer input
		for (i = 0; i < num_Class; i++) {
			output[i] = 0.0f;
			for (j = 0; j < inputDim; j++)
				output[i] += weight[i * inputDim + j] * x[j];
			output[i] += bias[i];
		}
		for (i = 0; i < num_Class; i++)
			output[i] = tanh(output[i]);

		//Find Maximum in neuron
		float temp = output[0];
		index_Max = 0;
		for (i = 1; i < num_Class; i++)
			if (temp < output[i]) {
				temp = output[i];
				index_Max = i;
			}

		delete[] output;
	}
	else {
		float output = 0.0f;
		for (j = 0; j < inputDim; j++)
			output += weight[j] * x[j];
		output += bias[0];
		output = tanh(output);
		if (output > 0.0f)
			index_Max = 0;
		else index_Max = 1;
	}
	return index_Max;

}//
float sgn_net(float net){
	if (net >= 0)
		return 1.0;
	else
		return -1.0;
};
float* Z_Score_Norm(float* Samples, int numSample, int inputDim) {
	float* normSamples = new float[numSample * inputDim];
	float* mean = new float[inputDim];
	float* std = new float[inputDim];

	// 1. Ortalamalar� hesapla
	for (int j = 0; j < inputDim; j++) {
		mean[j] = 0.0f;
		for (int i = 0; i < numSample; i++) {
			mean[j] += Samples[i * inputDim + j];
		}
		mean[j] /= numSample;
	}

	// 2. Standart sapmalar� hesapla
	for (int j = 0; j < inputDim; j++) {
		std[j] = 0.0f;
		for (int i = 0; i < numSample; i++) {
			float diff = Samples[i * inputDim + j] - mean[j];
			std[j] += diff * diff;
		}
		std[j] = sqrt(std[j] / numSample);
		if (std[j] == 0) std[j] = 1.0f; // B�lme s�f�r olmas�n
	}

	// 3. Normalize et
	for (int i = 0; i < numSample; i++) {
		for (int j = 0; j < inputDim; j++) {
			normSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / std[j];
		}
	}

	// Bellek temizli�i
	delete[] mean;
	delete[] std;

	return normSamples;
}

// Multi-layer forward pass (for testing/prediction)
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// class_count = output layer size
int Test_Forward_MultiLayer(float* x, float** Weights, float** bias,
	int inputDim, int* neuron_count, int Layer, int class_count) {

	// Build layer_sizes array: [hidden1, hidden2, ..., output]
	int num_layers = Layer + 1;  // hidden layers + output layer
	int* layer_sizes = new int[num_layers];
	
	// Copy hidden layer sizes
	for (int i = 0; i < Layer; i++) {
		layer_sizes[i] = neuron_count[i];
	}
	// Output layer size
	layer_sizes[num_layers - 1] = class_count;

	// Allocate arrays for layer outputs
	float** layer_outputs = new float* [num_layers];
	for (int layer = 0; layer < num_layers; layer++) {
		layer_outputs[layer] = new float[layer_sizes[layer]];
	}

	// Forward propagation through all layers
	for (int layer = 0; layer < num_layers; layer++) {
		// Determine input for this layer
		float* input;
		int input_size;

		if (layer == 0) {
			// First layer: input is x
			input = x;
			input_size = inputDim;
		}
		else {
			// Hidden/output layers: input is previous layer's output
			input = layer_outputs[layer - 1];
			input_size = layer_sizes[layer - 1];
		}

		// Calculate output for each neuron in this layer
		for (int j = 0; j < layer_sizes[layer]; j++) {
			float net = 0.0f;

			// Weighted sum
			for (int i = 0; i < input_size; i++) {
				net += Weights[layer][j * input_size + i] * input[i];
			}
			net += bias[layer][j];

			// Activation function: tanh
			layer_outputs[layer][j] = tanh(net);
		}
	}

	// Find the neuron with maximum output in the output layer
	int output_layer = num_layers - 1;
	int output_size = layer_sizes[output_layer];

	if (output_size == 1) {
		// Binary classification: check if output > 0
		int predicted_class = (layer_outputs[output_layer][0] > 0.0f) ? 0 : 1;

		// Cleanup
		for (int layer = 0; layer < num_layers; layer++) {
			delete[] layer_outputs[layer];
		}
		delete[] layer_outputs;
		delete[] layer_sizes;

		return predicted_class;
	}
	else {
		// Multi-class classification: find max output
		float max_output = layer_outputs[output_layer][0];
		int predicted_class = 0;

		for (int j = 1; j < output_size; j++) {
			if (layer_outputs[output_layer][j] > max_output) {
				max_output = layer_outputs[output_layer][j];
				predicted_class = j;
			}
		}

		// Cleanup
		for (int layer = 0; layer < num_layers; layer++) {
			delete[] layer_outputs[layer];
		}
		delete[] layer_outputs;
		delete[] layer_sizes;

		return predicted_class;
	}
}

// Multi-layer forward pass (for regression testing/prediction)
float Test_Forward_MultiLayer_Regression(float* x, float** Weights, float** bias,
	int inputDim, int* neuron_count, int Layer) {

	// Build layer_sizes array: [hidden1, hidden2, ..., output=1]
	int num_layers = Layer + 1;  // hidden layers + output layer
	int* layer_sizes = new int[num_layers];

	// Copy hidden layer sizes
	for (int i = 0; i < Layer; i++) {
		layer_sizes[i] = neuron_count[i];
	}
	// Output layer size for regression: 1
	layer_sizes[num_layers - 1] = 1;

	// Allocate arrays for layer outputs
	float** layer_outputs = new float* [num_layers];
	for (int layer = 0; layer < num_layers; layer++) {
		layer_outputs[layer] = new float[layer_sizes[layer]];
	}

	// Forward pass through all layers
	for (int layer = 0; layer < num_layers; layer++) {
		int input_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
		int output_size = layer_sizes[layer];
		float* layer_input = (layer == 0) ? x : layer_outputs[layer - 1];

		// Calculate weighted sum + bias and apply activation
		for (int j = 0; j < output_size; j++) {
			float net = bias[layer][j];
			for (int i = 0; i < input_size; i++) {
				net += Weights[layer][j * input_size + i] * layer_input[i];
			}

			// Activation: tanh for hidden layers, linear for output
			if (layer < num_layers - 1) {
				layer_outputs[layer][j] = tanh(net);
			}
			else {
				// Output layer: linear activation for regression
				layer_outputs[layer][j] = net;
			}
		}
	}

	// Get output value (continuous)
	int output_layer = num_layers - 1;
	float output_value = layer_outputs[output_layer][0];

	// Cleanup
	for (int layer = 0; layer < num_layers; layer++) {
		delete[] layer_outputs[layer];
	}
	delete[] layer_outputs;
	delete[] layer_sizes;

	return output_value;
}