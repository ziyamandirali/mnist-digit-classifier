#pragma once

using namespace System;
using namespace System::IO;
using namespace System::Drawing;

// MNIST PNG image loader and dataset utilities
class MNISTLoader {
public:
    // Load a single 28x28 PNG image and convert to normalized float array (784 elements)
    // Returns: float array of size 784, normalized to [-1, 1] for tanh activation
    // Caller is responsible for deleting the returned array
    static float* LoadImage(System::String^ filePath);

    // Load MNIST training dataset
    // Parameters:
    //   basePath: Path to "MNIST dataset/mnist-png" folder
    //   samplesPerDigit: Number of samples to load per digit (e.g., 100)
    //   Samples: Output pointer to samples array (will be allocated)
    //   targets: Output pointer to one-hot encoded targets (will be allocated)
    //   numSample: Output number of samples loaded
    static void LoadTrainDataset(System::String^ basePath, int samplesPerDigit,
        float*& Samples, float*& targets, int& numSample);

    // Load MNIST test dataset
    // Parameters:
    //   basePath: Path to "MNIST dataset/mnist-png" folder
    //   samplesPerDigit: Number of samples to load per digit (e.g., 10)
    //   Samples: Output pointer to samples array (will be allocated)
    //   targets: Output pointer to one-hot encoded targets (will be allocated)
    //   numSample: Output number of samples loaded
    static void LoadTestDataset(System::String^ basePath, int samplesPerDigit,
        float*& Samples, float*& targets, int& numSample);

private:
    // Select random files from a folder
    // Returns: Array of file paths
    static array<String^>^ SelectRandomFiles(System::String^ folderPath, int count);

    // Convert one-hot encoding to class label (0-9)
    static int GetClassLabel(float* oneHot, int numClasses);
};




