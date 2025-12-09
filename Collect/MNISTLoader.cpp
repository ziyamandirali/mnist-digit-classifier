#include "pch.h"
#include "MNISTLoader.h"
#include <algorithm>
#include <cstring>

using namespace System;
using namespace System::IO;
using namespace System::Drawing;

// Load a single 28x28 PNG image and convert to normalized float array
float* MNISTLoader::LoadImage(System::String^ filePath) {
    try {
        // Load image using System.Drawing
        Bitmap^ img = gcnew Bitmap(filePath);

        // Verify dimensions
        if (img->Width != 28 || img->Height != 28) {
            delete img;
            return nullptr;
        }

        // Allocate output array
        float* pixels = new float[784];

        // Convert to grayscale and normalize to [-1, 1]
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                Color pixel = img->GetPixel(x, y);
                // Since MNIST is grayscale, R=G=B
                int gray = pixel.R;
                // Normalize: [0, 255] -> [-1, 1] for tanh activation
                pixels[y * 28 + x] = (gray / 127.5f) - 1.0f;
            }
        }

        delete img;
        return pixels;
    }
    catch (Exception^ ex) {
        // Return nullptr on error
        return nullptr;
    }
}

// Select random files from a folder
array<String^>^ MNISTLoader::SelectRandomFiles(System::String^ folderPath, int count) {
    // Get all PNG files in the folder
    array<String^>^ allFiles = Directory::GetFiles(folderPath, "*.png");

    // If fewer files than requested, return all
    if (allFiles->Length <= count) {
        return allFiles;
    }

    // Create a copy for shuffling
    array<String^>^ filesCopy = gcnew array<String^>(allFiles->Length);
    Array::Copy(allFiles, filesCopy, allFiles->Length);

    // Fisher-Yates shuffle
    Random^ rng = gcnew Random();
    for (int i = filesCopy->Length - 1; i > 0; i--) {
        int j = rng->Next(i + 1);
        String^ temp = filesCopy[i];
        filesCopy[i] = filesCopy[j];
        filesCopy[j] = temp;
    }

    // Select first 'count' files
    array<String^>^ selected = gcnew array<String^>(count);
    Array::Copy(filesCopy, selected, count);

    return selected;
}

// Load MNIST training dataset
void MNISTLoader::LoadTrainDataset(System::String^ basePath, int samplesPerDigit,
    float*& Samples, float*& targets, int& numSample) {

    // Total samples: 10 digits × samplesPerDigit
    numSample = 10 * samplesPerDigit;
    int inputDim = 784;  // 28×28
    int numClasses = 10;

    // Allocate memory
    Samples = new float[numSample * inputDim];
    targets = new float[numSample * numClasses];

    // Initialize to zero
    memset(Samples, 0, numSample * inputDim * sizeof(float));
    memset(targets, 0, numSample * numClasses * sizeof(float));

	int sampleIndex = 0;

	// Load samples for each digit (0-9)
	for (int digit = 0; digit <= 9; digit++) {
		// Build folder path: basePath/train/digit/
		System::String^ folderPath = System::IO::Path::Combine(basePath, "train");
		folderPath = System::IO::Path::Combine(folderPath, digit.ToString());

		// Check if folder exists
		if (!Directory::Exists(folderPath)) {
			System::Diagnostics::Debug::WriteLine("Folder not found: " + folderPath);
			continue;
		}

		// Select random files
		array<String^>^ selectedFiles = SelectRandomFiles(folderPath, samplesPerDigit);
		System::Diagnostics::Debug::WriteLine("Digit " + digit + ": Selected " + selectedFiles->Length + " files");

		// Load each selected file
		for each (String^ filePath in selectedFiles) {
			// Load image as 784 float array
			float* pixels = LoadImage(filePath);

			if (pixels != nullptr) {
				// Copy pixels to Samples array
				memcpy(&Samples[sampleIndex * inputDim], pixels, inputDim * sizeof(float));
				delete[] pixels;

				// Set one-hot target (e.g., digit 5 → [0,0,0,0,0,1,0,0,0,0])
				targets[sampleIndex * numClasses + digit] = 1.0f;

				sampleIndex++;
			}
			else {
				System::Diagnostics::Debug::WriteLine("Failed to load: " + filePath);
			}
		}
	}

	// Update actual number of samples loaded (in case some failed)
	numSample = sampleIndex;
	System::Diagnostics::Debug::WriteLine("Total training samples loaded: " + numSample);
}

// Load MNIST test dataset
void MNISTLoader::LoadTestDataset(System::String^ basePath, int samplesPerDigit,
    float*& Samples, float*& targets, int& numSample) {

    // Total samples: 10 digits × samplesPerDigit
    numSample = 10 * samplesPerDigit;
    int inputDim = 784;  // 28×28
    int numClasses = 10;

    // Allocate memory
    Samples = new float[numSample * inputDim];
    targets = new float[numSample * numClasses];

    // Initialize to zero
    memset(Samples, 0, numSample * inputDim * sizeof(float));
    memset(targets, 0, numSample * numClasses * sizeof(float));

	int sampleIndex = 0;

	// Load samples for each digit (0-9)
	for (int digit = 0; digit <= 9; digit++) {
		// Build folder path: basePath/test/digit/
		System::String^ folderPath = System::IO::Path::Combine(basePath, "test");
		folderPath = System::IO::Path::Combine(folderPath, digit.ToString());

		// Check if folder exists
		if (!Directory::Exists(folderPath)) {
			System::Diagnostics::Debug::WriteLine("Folder not found: " + folderPath);
			continue;
		}

		// Select random files
		array<String^>^ selectedFiles = SelectRandomFiles(folderPath, samplesPerDigit);
		System::Diagnostics::Debug::WriteLine("Digit " + digit + ": Selected " + selectedFiles->Length + " files");

		// Load each selected file
		for each (String^ filePath in selectedFiles) {
			// Load image as 784 float array
			float* pixels = LoadImage(filePath);

			if (pixels != nullptr) {
				// Copy pixels to Samples array
				memcpy(&Samples[sampleIndex * inputDim], pixels, inputDim * sizeof(float));
				delete[] pixels;

				// Set one-hot target
				targets[sampleIndex * numClasses + digit] = 1.0f;

				sampleIndex++;
			}
			else {
				System::Diagnostics::Debug::WriteLine("Failed to load: " + filePath);
			}
		}
	}

	// Update actual number of samples loaded
	numSample = sampleIndex;
	System::Diagnostics::Debug::WriteLine("Total test samples loaded: " + numSample);
}

// Convert one-hot encoding to class label
int MNISTLoader::GetClassLabel(float* oneHot, int numClasses) {
    int maxIndex = 0;
    float maxValue = oneHot[0];

    for (int i = 1; i < numClasses; i++) {
        if (oneHot[i] > maxValue) {
            maxValue = oneHot[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

