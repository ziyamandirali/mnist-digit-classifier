#include "pch.h"
#include "Network.h"
#include "Process.h"
#include <cmath>

float* train_fcn(float* Samples, int numSample, float* targets, int inputDim, int class_count, float* Weights, float* bias, float learning_rate, float Min_Err, int Max_epoch, int& epoch) {

	float total_err;
	float* temp = new float[Max_epoch];
	float* net, * fnet, * f_der, * desired, * err, * delta;
	net = new float[class_count];
	fnet = new float[class_count];
	f_der = new float[class_count];
	desired = new float[class_count];
	err = new float[class_count];
	delta = new float[class_count];
	epoch = 0;

	do {
		total_err = 0.0f;
		for (int step = 0; step < numSample; step++) {
			//FeedForward
			//FeedForward
			for (int k = 0; k < class_count; k++) {
				net[k] = 0.0f;
				for (int i = 0; i < inputDim; i++) {
					net[k] += Weights[k * inputDim + i] * Samples[step * inputDim + i];
				}
				net[k] += bias[k];
				fnet[k] = (2.0f / (float)(1.0f + exp(-net[k])) - 1.0f);
				f_der[k] = 0.5f * (1.0f - fnet[k] * fnet[k]);
			}

			//Backward
			for (int k = 0; k < class_count; k++) {
				if (targets[step] == k) {
					desired[k] = 1.0;
				}
				else desired[k] = -1.0;
				
				err[k] = desired[k] - fnet[k];
				delta[k] = learning_rate * err[k] * f_der[k];
				
				// Gradient clipping to prevent overflow
				if (delta[k] > 10.0f) delta[k] = 10.0f;
				if (delta[k] < -10.0f) delta[k] = -10.0f;
				
				for (int i = 0; i < inputDim; i++) {
					Weights[k * inputDim + i] += delta[k] * Samples[step * inputDim + i];
					// Check for NaN/Inf
					if (!isfinite(Weights[k * inputDim + i])) {
						Weights[k * inputDim + i] = 0.0f;
					}
				}
				bias[k] += delta[k];
				// Check for NaN/Inf in bias
				if (!isfinite(bias[k])) {
					bias[k] = 0.0f;
				}
				total_err += (0.5f * (err[k] * err[k]));
			}
		}
		total_err /= float(numSample);
		if (epoch < Max_epoch) {
			temp[epoch] = total_err;
			epoch++;
			
			// Progress reporting: Print every 10% of epochs
			int progress_interval = Max_epoch / 10;
			if (progress_interval == 0) progress_interval = 1;
			
			if (epoch % progress_interval == 0 || epoch == 1) {
				float progress_percent = (float)epoch / Max_epoch * 100.0f;
				System::String^ msg = System::String::Format(
					"[SINGLE-LAYER] Epoch {0}/{1} ({2:F1}%) | Error: {3:F6}",
					epoch, Max_epoch, progress_percent, total_err);
				System::Diagnostics::Debug::WriteLine(msg);
			}
		}
	} while ((total_err > Min_Err) && (epoch < Max_epoch));
	delete[] net;
	delete[] fnet;
	delete[] f_der;
	delete[] desired;
	delete[] err;
	delete[] delta;
	return temp;
} //train

// Multi-layer training with backpropagation
// neuron_count = array of hidden layer sizes [h1_size, h2_size, ...]
// Layer = number of hidden layers
// class_count = output layer size
float* train_fcn_multilayer(float* Samples, int numSample, float* targets,
	int inputDim, int* neuron_count, int Layer, int class_count,
	float** Weights, float** bias,
	float learning_rate, float Min_Err, int Max_epoch, int& epoch,
	float momentum) {

	float total_err;
	float* error_history = new float[Max_epoch];
	epoch = 0;

	// Build layer_sizes array: [hidden1, hidden2, ..., output]
	int num_layers = Layer + 1;  // hidden layers + output layer
	int* layer_sizes = new int[num_layers];
	
	// Copy hidden layer sizes
	for (int i = 0; i < Layer; i++) {
		layer_sizes[i] = neuron_count[i];
	}
	// Output layer size
	layer_sizes[num_layers - 1] = class_count;

	// Allocate arrays for forward pass (store activations for each layer)
	float** layer_outputs = new float* [num_layers];
	float** layer_nets = new float* [num_layers];
	float** layer_derivatives = new float* [num_layers];
	for (int layer = 0; layer < num_layers; layer++) {
		layer_outputs[layer] = new float[layer_sizes[layer]];
		layer_nets[layer] = new float[layer_sizes[layer]];
		layer_derivatives[layer] = new float[layer_sizes[layer]];
	}

	// Allocate arrays for backward pass (deltas for each layer)
	float** deltas = new float* [num_layers];
	for (int layer = 0; layer < num_layers; layer++) {
		deltas[layer] = new float[layer_sizes[layer]];
	}

	// Allocate velocity arrays for momentum (initialized to 0)
	float*** velocity_weights = nullptr;
	float** velocity_bias = nullptr;
	if (momentum > 0.0f) {
		velocity_weights = new float** [num_layers];
		velocity_bias = new float* [num_layers];
		
		for (int layer = 0; layer < num_layers; layer++) {
			int input_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
			int output_size = layer_sizes[layer];
			
			velocity_weights[layer] = new float*[output_size];
			velocity_bias[layer] = new float[output_size];
			
			for (int j = 0; j < output_size; j++) {
				velocity_weights[layer][j] = new float[input_size];
				// Initialize to zero
				for (int i = 0; i < input_size; i++) {
					velocity_weights[layer][j][i] = 0.0f;
				}
				velocity_bias[layer][j] = 0.0f;
			}
		}
	}

	// Desired output array (for output layer)
	int output_size = layer_sizes[num_layers - 1];
	float* desired = new float[output_size];

	// Create shuffle indices array
	int* shuffle_indices = new int[numSample];
	for (int i = 0; i < numSample; i++) {
		shuffle_indices[i] = i;
	}

	do {
		total_err = 0.0f;

		// Shuffle training data every epoch (Fisher-Yates shuffle)
		for (int i = numSample - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			// Swap indices
			int temp = shuffle_indices[i];
			shuffle_indices[i] = shuffle_indices[j];
			shuffle_indices[j] = temp;
		}

		for (int step = 0; step < numSample; step++) {
			// Use shuffled index
			int sample_idx = shuffle_indices[step];
			// ===== FORWARD PROPAGATION =====
			for (int layer = 0; layer < num_layers; layer++) {
				// Determine input for this layer
				float* input;
				int input_size;

			if (layer == 0) {
				// First layer: input is the sample (use shuffled index)
				input = &Samples[sample_idx * inputDim];
				input_size = inputDim;
			}
				else {
					// Hidden/output layers: input is previous layer's output
					input = layer_outputs[layer - 1];
					input_size = layer_sizes[layer - 1];
				}

				// Calculate net and activation for each neuron in this layer
				for (int j = 0; j < layer_sizes[layer]; j++) {
					layer_nets[layer][j] = 0.0f;

					// Weighted sum
					for (int i = 0; i < input_size; i++) {
						layer_nets[layer][j] += Weights[layer][j * input_size + i] * input[i];
					}
					layer_nets[layer][j] += bias[layer][j];

					// Activation function: tanh
					layer_outputs[layer][j] = tanh(layer_nets[layer][j]);

					// Derivative of tanh: 1 - tanh^2
					layer_derivatives[layer][j] = 1.0f - layer_outputs[layer][j] * layer_outputs[layer][j];
				}
			}

			// ===== BACKWARD PROPAGATION =====
			
			// Output layer error
			int output_layer = num_layers - 1;
			for (int j = 0; j < output_size; j++) {
			// Desired output (one-hot or bipolar encoding)
			if (output_size == 1) {
				// Binary classification
				desired[j] = (targets[sample_idx] == 0) ? 1.0f : -1.0f;
			}
			else {
				// Multi-class classification (targets are one-hot encoded)
				// For MNIST: targets[sample_idx * class_count + j] gives the target for class j
				desired[j] = targets[sample_idx * output_size + j];
			}

				// Calculate error
				float error = desired[j] - layer_outputs[output_layer][j];
				total_err += 0.5f * error * error;

				// Delta for output layer: error * derivative
				deltas[output_layer][j] = error * layer_derivatives[output_layer][j];

				// Gradient clipping
				if (deltas[output_layer][j] > 10.0f) deltas[output_layer][j] = 10.0f;
				if (deltas[output_layer][j] < -10.0f) deltas[output_layer][j] = -10.0f;
			}

			// Hidden layers error (backpropagate)
			for (int layer = num_layers - 2; layer >= 0; layer--) {
				for (int j = 0; j < layer_sizes[layer]; j++) {
					float error_sum = 0.0f;

					// Sum weighted deltas from next layer
					int next_layer = layer + 1;
					for (int k = 0; k < layer_sizes[next_layer]; k++) {
						error_sum += deltas[next_layer][k] * Weights[next_layer][k * layer_sizes[layer] + j];
					}

					// Delta for hidden layer: backpropagated error * derivative
					deltas[layer][j] = error_sum * layer_derivatives[layer][j];

					// Gradient clipping
					if (deltas[layer][j] > 10.0f) deltas[layer][j] = 10.0f;
					if (deltas[layer][j] < -10.0f) deltas[layer][j] = -10.0f;
				}
			}

			// ===== UPDATE WEIGHTS AND BIASES =====
			for (int layer = 0; layer < num_layers; layer++) {
				// Determine input for this layer
				float* input;
				int input_size;

				if (layer == 0) {
					input = &Samples[sample_idx * inputDim];  // FIX: Use shuffled index!
					input_size = inputDim;
				}
				else {
					input = layer_outputs[layer - 1];
					input_size = layer_sizes[layer - 1];
				}

			// Update weights and biases
			for (int j = 0; j < layer_sizes[layer]; j++) {
				if (momentum > 0.0f) {
					// ===== WITH MOMENTUM =====
					// Update weights
					for (int i = 0; i < input_size; i++) {
						// Calculate gradient
						float gradient = deltas[layer][j] * input[i];
						
						// Update velocity: v = momentum * v + gradient
						velocity_weights[layer][j][i] = momentum * velocity_weights[layer][j][i] + gradient;
						
						// Update weight: w = w + learning_rate * v
						Weights[layer][j * input_size + i] += learning_rate * velocity_weights[layer][j][i];

						// Check for NaN/Inf
						if (!isfinite(Weights[layer][j * input_size + i])) {
							Weights[layer][j * input_size + i] = 0.0f;
							velocity_weights[layer][j][i] = 0.0f;
						}
					}

					// Update bias with momentum
					velocity_bias[layer][j] = momentum * velocity_bias[layer][j] + deltas[layer][j];
					bias[layer][j] += learning_rate * velocity_bias[layer][j];

					// Check for NaN/Inf
					if (!isfinite(bias[layer][j])) {
						bias[layer][j] = 0.0f;
						velocity_bias[layer][j] = 0.0f;
					}
				}
				else {
					// ===== WITHOUT MOMENTUM (Original) =====
					// Update weights
					for (int i = 0; i < input_size; i++) {
						Weights[layer][j * input_size + i] += learning_rate * deltas[layer][j] * input[i];

						// Check for NaN/Inf
						if (!isfinite(Weights[layer][j * input_size + i])) {
							Weights[layer][j * input_size + i] = 0.0f;
						}
					}

					// Update bias
					bias[layer][j] += learning_rate * deltas[layer][j];

					// Check for NaN/Inf
					if (!isfinite(bias[layer][j])) {
						bias[layer][j] = 0.0f;
					}
				}
			}
			}
		}

		// Average error over all samples
		total_err /= float(numSample);

		if (epoch < Max_epoch) {
			error_history[epoch] = total_err;
			epoch++;
			
			// Progress reporting: Print every 10% of epochs
			int progress_interval = Max_epoch / 10;
			if (progress_interval == 0) progress_interval = 1;
			
			if (epoch % progress_interval == 0 || epoch == 1) {
				float progress_percent = (float)epoch / Max_epoch * 100.0f;
				System::String^ msg = System::String::Format(
					"Epoch {0}/{1} ({2:F1}%) | Error: {3:F6}",
					epoch, Max_epoch, progress_percent, total_err);
				System::Diagnostics::Debug::WriteLine(msg);
			}
		}

	} while ((total_err > Min_Err) && (epoch < Max_epoch));

	// Cleanup momentum arrays (before layer_sizes is deleted)
	if (momentum > 0.0f && velocity_weights && velocity_bias) {
		for (int layer = 0; layer < num_layers; layer++) {
			int output_size = layer_sizes[layer];
			for (int j = 0; j < output_size; j++) {
				delete[] velocity_weights[layer][j];
			}
			delete[] velocity_weights[layer];
			delete[] velocity_bias[layer];
		}
		delete[] velocity_weights;
		delete[] velocity_bias;
	}
	
	// Cleanup
	for (int layer = 0; layer < num_layers; layer++) {
		delete[] layer_outputs[layer];
		delete[] layer_nets[layer];
		delete[] layer_derivatives[layer];
		delete[] deltas[layer];
	}
	delete[] layer_outputs;
	delete[] layer_nets;
	delete[] layer_derivatives;
	delete[] deltas;
	delete[] desired;
	delete[] layer_sizes;
	delete[] shuffle_indices;

	return error_history;
} //train_fcn_multilayer

// Multi-layer regression training
float* train_fcn_multilayer_regression(float* Samples, int numSample, float* targets,
	int inputDim, int* neuron_count, int Layer,
	float** Weights, float** bias,
	float learning_rate, float Min_Err, int Max_epoch, int& epoch,
	float momentum) {

	// Build full layer structure: [hidden1, hidden2, ..., output=1]
	int num_layers = Layer + 1;  // hidden layers + output layer
	int* layer_sizes = new int[num_layers];

	// Copy hidden layer sizes
	for (int i = 0; i < Layer; i++) {
		layer_sizes[i] = neuron_count[i];
	}
	// Output layer for regression: 1 neuron
	layer_sizes[num_layers - 1] = 1;

	// Allocate arrays for each layer
	float** layer_outputs = new float* [num_layers];
	float** layer_nets = new float* [num_layers];
	float** layer_derivatives = new float* [num_layers];
	float** deltas = new float* [num_layers];

	for (int layer = 0; layer < num_layers; layer++) {
		layer_outputs[layer] = new float[layer_sizes[layer]];
		layer_nets[layer] = new float[layer_sizes[layer]];
		layer_derivatives[layer] = new float[layer_sizes[layer]];
		deltas[layer] = new float[layer_sizes[layer]];
	}

	// Allocate velocity arrays for momentum (initialized to 0)
	float*** velocity_weights = nullptr;
	float** velocity_bias = nullptr;
	if (momentum > 0.0f) {
		velocity_weights = new float** [num_layers];
		velocity_bias = new float* [num_layers];
		
		for (int layer = 0; layer < num_layers; layer++) {
			int input_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
			int output_size = layer_sizes[layer];
			
			velocity_weights[layer] = new float*[output_size];
			velocity_bias[layer] = new float[output_size];
			
			for (int j = 0; j < output_size; j++) {
				velocity_weights[layer][j] = new float[input_size];
				// Initialize to zero
				for (int i = 0; i < input_size; i++) {
					velocity_weights[layer][j][i] = 0.0f;
				}
				velocity_bias[layer][j] = 0.0f;
			}
		}
	}

	float* error_history = new float[Max_epoch];
	const float gradient_clip = 5.0f;

	// Training loop
	for (epoch = 0; epoch < Max_epoch; epoch++) {
		float total_error = 0.0f;

		for (int sample_idx = 0; sample_idx < numSample; sample_idx++) {
			float* current_sample = &Samples[sample_idx * inputDim];
			float target_value = targets[sample_idx];

			// ===== FORWARD PASS =====
			for (int layer = 0; layer < num_layers; layer++) {
				int input_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
				int output_size = layer_sizes[layer];
				float* layer_input = (layer == 0) ? current_sample : layer_outputs[layer - 1];

				// Calculate weighted sum + bias
				for (int j = 0; j < output_size; j++) {
					float net = bias[layer][j];
					for (int i = 0; i < input_size; i++) {
						net += Weights[layer][j * input_size + i] * layer_input[i];
					}
					layer_nets[layer][j] = net;

					// Activation: tanh for hidden, linear for output
					if (layer < num_layers - 1) {
						layer_outputs[layer][j] = tanh(net);
						layer_derivatives[layer][j] = 1.0f - layer_outputs[layer][j] * layer_outputs[layer][j];
					}
					else {
						// Output layer: linear activation for regression
						layer_outputs[layer][j] = net;
						layer_derivatives[layer][j] = 1.0f;  // derivative of linear is 1
					}
				}
			}

			// ===== CALCULATE ERROR (MSE) =====
			int output_layer = num_layers - 1;
			float output_value = layer_outputs[output_layer][0];
			float error = target_value - output_value;
			total_error += error * error;  // Squared error

			// ===== BACKWARD PASS =====
			// Output layer delta (for regression with linear output)
			deltas[output_layer][0] = error * layer_derivatives[output_layer][0];

			// Check for NaN/Inf
			if (std::isnan(deltas[output_layer][0]) || std::isinf(deltas[output_layer][0])) {
				deltas[output_layer][0] = 0.0f;
			}

			// Backpropagate to hidden layers
			for (int layer = num_layers - 2; layer >= 0; layer--) {
				int current_size = layer_sizes[layer];
				int next_size = layer_sizes[layer + 1];

				for (int j = 0; j < current_size; j++) {
					float error_sum = 0.0f;
					for (int k = 0; k < next_size; k++) {
						error_sum += deltas[layer + 1][k] * Weights[layer + 1][k * current_size + j];
					}
					deltas[layer][j] = error_sum * layer_derivatives[layer][j];

					// Check for NaN/Inf
					if (std::isnan(deltas[layer][j]) || std::isinf(deltas[layer][j])) {
						deltas[layer][j] = 0.0f;
					}
					// Gradient clipping
					if (deltas[layer][j] > gradient_clip) deltas[layer][j] = gradient_clip;
					if (deltas[layer][j] < -gradient_clip) deltas[layer][j] = -gradient_clip;
				}
			}

			// ===== UPDATE WEIGHTS AND BIASES =====
			for (int layer = 0; layer < num_layers; layer++) {
				int input_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
				int output_size = layer_sizes[layer];
				float* layer_input = (layer == 0) ? current_sample : layer_outputs[layer - 1];

			for (int j = 0; j < output_size; j++) {
				if (momentum > 0.0f) {
					// ===== WITH MOMENTUM =====
					// Update bias with momentum
					velocity_bias[layer][j] = momentum * velocity_bias[layer][j] + deltas[layer][j];
					bias[layer][j] += learning_rate * velocity_bias[layer][j];

					// Check bias for NaN/Inf
					if (std::isnan(bias[layer][j]) || std::isinf(bias[layer][j])) {
						bias[layer][j] = 0.01f;
						velocity_bias[layer][j] = 0.0f;
					}

					// Update weights with momentum
					for (int i = 0; i < input_size; i++) {
						float gradient = deltas[layer][j] * layer_input[i];

						// Gradient clipping
						if (gradient > gradient_clip) gradient = gradient_clip;
						if (gradient < -gradient_clip) gradient = -gradient_clip;

						// Update velocity: v = momentum * v + gradient
						velocity_weights[layer][j][i] = momentum * velocity_weights[layer][j][i] + gradient;
						
						// Update weight: w = w + learning_rate * v
						Weights[layer][j * input_size + i] += learning_rate * velocity_weights[layer][j][i];

						// Check weights for NaN/Inf
						if (std::isnan(Weights[layer][j * input_size + i]) ||
							std::isinf(Weights[layer][j * input_size + i])) {
							Weights[layer][j * input_size + i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
							velocity_weights[layer][j][i] = 0.0f;
						}
					}
				}
				else {
					// ===== WITHOUT MOMENTUM (Original) =====
					// Update bias
					bias[layer][j] += learning_rate * deltas[layer][j];

					// Check bias for NaN/Inf
					if (std::isnan(bias[layer][j]) || std::isinf(bias[layer][j])) {
						bias[layer][j] = 0.01f;
					}

					// Update weights
					for (int i = 0; i < input_size; i++) {
						float weight_update = learning_rate * deltas[layer][j] * layer_input[i];

						// Gradient clipping
						if (weight_update > gradient_clip) weight_update = gradient_clip;
						if (weight_update < -gradient_clip) weight_update = -gradient_clip;

						Weights[layer][j * input_size + i] += weight_update;

						// Check weights for NaN/Inf
						if (std::isnan(Weights[layer][j * input_size + i]) ||
							std::isinf(Weights[layer][j * input_size + i])) {
							Weights[layer][j * input_size + i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
						}
					}
				}
			}
			}
		} // End of sample loop

		// Calculate MSE
		float mse = total_error / numSample;
		error_history[epoch] = mse;

		// Progress reporting: Print every 10% of epochs
		int progress_interval = Max_epoch / 10;
		if (progress_interval == 0) progress_interval = 1;
		
		if ((epoch + 1) % progress_interval == 0 || epoch == 0) {
			float progress_percent = (float)(epoch + 1) / Max_epoch * 100.0f;
			System::String^ msg = System::String::Format(
				"[REGRESSION] Epoch {0}/{1} ({2:F1}%) | Error: {3:F6}",
				epoch + 1, Max_epoch, progress_percent, mse);
			System::Diagnostics::Debug::WriteLine(msg);
		}

		// Check convergence
		if (mse < Min_Err) {
			epoch++;
			break;
		}
	} // End of epoch loop

	// Cleanup momentum arrays (before layer_sizes is deleted)
	if (momentum > 0.0f && velocity_weights && velocity_bias) {
		for (int layer = 0; layer < num_layers; layer++) {
			int output_size = layer_sizes[layer];
			for (int j = 0; j < output_size; j++) {
				delete[] velocity_weights[layer][j];
			}
			delete[] velocity_weights[layer];
			delete[] velocity_bias[layer];
		}
		delete[] velocity_weights;
		delete[] velocity_bias;
	}
	
	// Cleanup
	for (int layer = 0; layer < num_layers; layer++) {
		delete[] layer_outputs[layer];
		delete[] layer_nets[layer];
		delete[] layer_derivatives[layer];
		delete[] deltas[layer];
	}
	delete[] layer_outputs;
	delete[] layer_nets;
	delete[] layer_derivatives;
	delete[] deltas;
	delete[] layer_sizes;

	return error_history;
} //train_fcn_multilayer_regression

// Single-layer Linear Regression Training
float* regression_train(float* x, float* y, int numSample, float& slope, float& intercept, 
                        float learning_rate, float Min_Err, int Max_epoch, int& epoch) {
	
	float* error_history = new float[Max_epoch];
	float total_err = 0.0f;
	epoch = 0;
	
	// Initialize parameters
	slope = 0.0f;
	intercept = 0.0f;
	
	do {
		total_err = 0.0f;
		float slope_gradient = 0.0f;
		float intercept_gradient = 0.0f;
		
		// Calculate gradients
		for (int i = 0; i < numSample; i++) {
			float y_pred = slope * x[i] + intercept;
			float error = y[i] - y_pred;
			
			slope_gradient += -2.0f * error * x[i];
			intercept_gradient += -2.0f * error;
			
			total_err += (error * error);
		}
		
		// Average gradients and error
		slope_gradient /= numSample;
		intercept_gradient /= numSample;
		total_err /= numSample;
		
		// Update parameters
		slope -= learning_rate * slope_gradient;
		intercept -= learning_rate * intercept_gradient;
		
		// Gradient clipping
		if (slope > 100.0f) slope = 100.0f;
		if (slope < -100.0f) slope = -100.0f;
		if (intercept > 1000.0f) intercept = 1000.0f;
		if (intercept < -1000.0f) intercept = -1000.0f;
		
		// Check for NaN/Inf
		if (!isfinite(slope)) slope = 0.0f;
		if (!isfinite(intercept)) intercept = 0.0f;
		
		if (epoch < Max_epoch) {
			error_history[epoch] = total_err;
			epoch++;
			
			// Progress reporting: Print every 10% of epochs
			int progress_interval = Max_epoch / 10;
			if (progress_interval == 0) progress_interval = 1;
			
			if (epoch % progress_interval == 0 || epoch == 1) {
				float progress_percent = (float)epoch / Max_epoch * 100.0f;
				System::String^ msg = System::String::Format(
					"[LINEAR REGRESSION] Epoch {0}/{1} ({2:F1}%) | Error: {3:F6}",
					epoch, Max_epoch, progress_percent, total_err);
				System::Diagnostics::Debug::WriteLine(msg);
			}
		}
		
	} while ((total_err > Min_Err) && (epoch < Max_epoch));
	
	return error_history;
} //regression_train

