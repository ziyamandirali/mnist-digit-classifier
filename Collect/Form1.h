#pragma once
#include "Process.h"
#include "Network.h"
#include "MNISTLoader.h"
#include "TrainingDialog.h"
#include <iostream>
#include <fstream>
#include <string>

namespace CppCLRWinformsProjekt {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;
	using namespace System::IO;

	/// <summary>
	/// Zusammenfassung fr Form1
	/// </summary>
	public ref class Form1 : public System::Windows::Forms::Form
	{
	public:
		Form1(void)
		{
			InitializeComponent();
			//
			//TODO: Konstruktorcode hier hinzufgen.
			//
		neuron_count = 0;
		mean = nullptr;
		std = nullptr;
		neuron_count = 0;
		mean = nullptr;
		std = nullptr;
		// regression_slope = 0.0f; // Removed
		// regression_intercept = 0.0f; // Removed
		regression_trained = false;
		regression_is_multilayer = false;
		regression_mean = nullptr;
		regression_std = nullptr;
		
		// Multi-layer initialization
		num_layers = 0;
		layer_sizes = nullptr;
		Weights_ML = nullptr;
		bias_ML = nullptr;
		is_multilayer = false;

		// Autoencoder initialization
		autoencoder_weights = nullptr;
		autoencoder_bias = nullptr;
		autoencoder_layer_sizes = nullptr;
		autoencoder_num_layers = 0;
		autoencoder_latent_dim = 0;
		autoencoder_trained = false;
		encoder_weights = nullptr;
		encoder_bias = nullptr;

		// Encoder-based classifier initialization
		encoder_classifier_weights = nullptr;
		encoder_classifier_bias = nullptr;
		encoder_classifier_layers = nullptr;
		encoder_classifier_num_layers = 0;
		encoder_classifier_trained = false;
	}

	protected:
		/// <summary>
		/// Verwendete Ressourcen bereinigen.
		/// </summary>
		~Form1()
		{
			if (components)
			{
				delete components;
			}
			// Dinamik dizileri güvenli şekilde temizle
			Cleanup_Samples();
			Cleanup_Network();
			Cleanup_Normalization();
	// MNIST cleanup
	if (mnist_train_samples) {
		delete[] mnist_train_samples;
		mnist_train_samples = nullptr;
	}
	if (mnist_train_targets) {
		delete[] mnist_train_targets;
		mnist_train_targets = nullptr;
	}
	if (mnist_test_samples) {
		delete[] mnist_test_samples;
		mnist_test_samples = nullptr;
	}
	if (mnist_test_targets) {
		delete[] mnist_test_targets;
		mnist_test_targets = nullptr;
	}
	// MNIST weights cleanup
	if (mnist_weights) {
		int total_layers = mnist_hidden_layers + 1;
		for (int i = 0; i < total_layers; i++) {
			if (mnist_weights[i]) {
				delete[] mnist_weights[i];
				mnist_weights[i] = nullptr;
			}
		}
		delete[] mnist_weights;
		mnist_weights = nullptr;
	}
	if (mnist_bias) {
		int total_layers = mnist_hidden_layers + 1;
		for (int i = 0; i < total_layers; i++) {
			if (mnist_bias[i]) {
				delete[] mnist_bias[i];
				mnist_bias[i] = nullptr;
			}
		}
		delete[] mnist_bias;
		mnist_bias = nullptr;
	}
	if (mnist_layer_sizes) {
		delete[] mnist_layer_sizes;
		mnist_layer_sizes = nullptr;
	}
	
	// Autoencoder cleanup
	if (autoencoder_weights) {
		for (int i = 0; i < autoencoder_num_layers; i++) {
			if (autoencoder_weights[i]) {
				delete[] autoencoder_weights[i];
				autoencoder_weights[i] = nullptr;
			}
		}
		delete[] autoencoder_weights;
		autoencoder_weights = nullptr;
	}
	if (autoencoder_bias) {
		for (int i = 0; i < autoencoder_num_layers; i++) {
			if (autoencoder_bias[i]) {
				delete[] autoencoder_bias[i];
				autoencoder_bias[i] = nullptr;
			}
		}
		delete[] autoencoder_bias;
		autoencoder_bias = nullptr;
	}
	if (autoencoder_layer_sizes) {
		delete[] autoencoder_layer_sizes;
		autoencoder_layer_sizes = nullptr;
	}
	if (encoder_weights) {
		int encoder_layers = autoencoder_num_layers / 2;
		for (int i = 0; i < encoder_layers; i++) {
			if (encoder_weights[i]) {
				delete[] encoder_weights[i];
				encoder_weights[i] = nullptr;
			}
		}
		delete[] encoder_weights;
		encoder_weights = nullptr;
	}
	if (encoder_bias) {
		int encoder_layers = autoencoder_num_layers / 2;
		for (int i = 0; i < encoder_layers; i++) {
			if (encoder_bias[i]) {
				delete[] encoder_bias[i];
				encoder_bias[i] = nullptr;
			}
		}
		delete[] encoder_bias;
		encoder_bias = nullptr;
	}
	
	// Encoder-based classifier cleanup
	if (encoder_classifier_weights) {
		for (int i = 0; i < encoder_classifier_num_layers; i++) {
			if (encoder_classifier_weights[i]) {
				delete[] encoder_classifier_weights[i];
				encoder_classifier_weights[i] = nullptr;
			}
		}
		delete[] encoder_classifier_weights;
		encoder_classifier_weights = nullptr;
	}
	if (encoder_classifier_bias) {
		for (int i = 0; i < encoder_classifier_num_layers; i++) {
			if (encoder_classifier_bias[i]) {
				delete[] encoder_classifier_bias[i];
				encoder_classifier_bias[i] = nullptr;
			}
		}
		delete[] encoder_classifier_bias;
		encoder_classifier_bias = nullptr;
	}
	if (encoder_classifier_layers) {
		delete[] encoder_classifier_layers;
		encoder_classifier_layers = nullptr;
	}
		// bias_ML and layer_sizes handled by Cleanup_Network above
	}  // End of ~Form1() destructor

	private: System::Windows::Forms::PictureBox^ pictureBox1;
	protected:
	private: System::Windows::Forms::GroupBox^ groupBox1;
	private: System::Windows::Forms::Button^ Set_Net;

	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::ComboBox^ ClassCountBox;

	private: System::Windows::Forms::GroupBox^ groupBox2;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::ComboBox^ ClassNoBox;

	private: System::Windows::Forms::Label^ label3;

	private: System::Windows::Forms::Label^ labelHiddenLayers;
	private: System::Windows::Forms::ComboBox^ HiddenLayerCountBox;
	private: System::Windows::Forms::Label^ labelLayer1;
	private: System::Windows::Forms::TextBox^ Layer1TextBox;
	private: System::Windows::Forms::Label^ labelLayer2;
	private: System::Windows::Forms::TextBox^ Layer2TextBox;
	private: System::Windows::Forms::Label^ labelLayer3;
	private: System::Windows::Forms::TextBox^ Layer3TextBox;
	private: System::Windows::Forms::Label^ labelLayer4;
	private: System::Windows::Forms::TextBox^ Layer4TextBox;
	private: System::Windows::Forms::Label^ labelLayer5;
	private: System::Windows::Forms::TextBox^ Layer5TextBox;
	private: System::Windows::Forms::Label^ labelLayer6;
	private: System::Windows::Forms::TextBox^ Layer6TextBox;
	private: System::Windows::Forms::Label^ labelLayer7;
	private: System::Windows::Forms::TextBox^ Layer7TextBox;
	private: System::Windows::Forms::Label^ labelLayer8;
	private: System::Windows::Forms::TextBox^ Layer8TextBox;
	private: System::Windows::Forms::Label^ labelLayer9;
	private: System::Windows::Forms::TextBox^ Layer9TextBox;
	private: System::Windows::Forms::Label^ labelLayer10;
	private: System::Windows::Forms::TextBox^ Layer10TextBox;

	private:
		/// <summary>
		/// User Defined Variables
		int  class_count = 0, numSample = 0, inputDim = 2;
		int  neuron_count = 0; // Actual number of output neurons (1 for binary, class_count for multi-class)
		float* Samples, * targets, * Weights, * bias;
		float* mean, * std; // Normalization parameters
		
		// Regression variables
		// float regression_slope = 0.0f; // Removed
		// float regression_intercept = 0.0f; // Removed
		bool regression_trained = false;
		bool regression_is_multilayer = false;  // Flag for regression type
		float* regression_mean = nullptr;       // Normalization for regression
		float* regression_std = nullptr;        // Normalization for regression

		// Multi-Layer variables
		int num_layers = 0;              // Total layers (hidden + output)
		int* layer_sizes = nullptr;     // Size of each layer
		float** Weights_ML = nullptr;    // Multi-layer weights
		float** bias_ML = nullptr;       // Multi-layer bias
		bool is_multilayer = false;      // Flag: single or multi-layer

	// MNIST variables
	float* mnist_train_samples = nullptr;
	float* mnist_train_targets = nullptr;
	int mnist_train_count = 0;
	float* mnist_test_samples = nullptr;
	float* mnist_test_targets = nullptr;
	int mnist_test_count = 0;
	bool mnist_loaded = false;
	System::String^ mnist_base_path = "MNIST dataset\\mnist-png";
	
	// MNIST network parameters (saved after training)
	float** mnist_weights = nullptr;
	float** mnist_bias = nullptr;
	int* mnist_layer_sizes = nullptr;
	int mnist_hidden_layers = 0;
	int mnist_input_dim = 0;
	int mnist_class_count = 0;
	bool mnist_trained = false;

	// Autoencoder variables
	float** autoencoder_weights = nullptr;  // Encoder + Decoder weights
	float** autoencoder_bias = nullptr;     // Encoder + Decoder bias
	int* autoencoder_layer_sizes = nullptr; // Layer sizes (excluding input/output)
	int autoencoder_num_layers = 0;         // Total layers (encoder + decoder)
	int autoencoder_latent_dim = 0;         // Latent space dimensionality
	bool autoencoder_trained = false;       // Training status
	float** encoder_weights = nullptr;      // Just encoder part (for feature extraction)
	float** encoder_bias = nullptr;         // Just encoder bias

	// Encoder-based classification variables
	float** encoder_classifier_weights = nullptr;  // Classification weights on encoded features
	float** encoder_classifier_bias = nullptr;     // Classification bias
	int* encoder_classifier_layers = nullptr;      // Hidden layers for classifier
	int encoder_classifier_num_layers = 0;         // Number of layers in classifier
	bool encoder_classifier_trained = false;       // Training status

	// Helper methods for memory cleanup
	private: void Cleanup_Samples() {
		if (Samples) { delete[] Samples; Samples = nullptr; }
		if (targets) { delete[] targets; targets = nullptr; }
		numSample = 0;
		if (label3) label3->Text = "Samples Count: 0";
	}
	
	private: void Cleanup_Normalization() {
		if (mean) { delete[] mean; mean = nullptr; }
		if (std) { delete[] std; std = nullptr; }
		if (regression_mean) { delete[] regression_mean; regression_mean = nullptr; }
		if (regression_std) { delete[] regression_std; regression_std = nullptr; }
	}
	
	private: void Cleanup_Network() {
		if (Weights) { delete[] Weights; Weights = nullptr; }
		if (bias) { delete[] bias; bias = nullptr; }
		
		if (Weights_ML) {
			for (int i = 0; i < num_layers; i++) {
				if (Weights_ML[i]) delete[] Weights_ML[i];
			}
			delete[] Weights_ML;
			Weights_ML = nullptr;
		}
		if (bias_ML) {
			for (int i = 0; i < num_layers; i++) {
				if (bias_ML[i]) delete[] bias_ML[i];
			}
			delete[] bias_ML;
			bias_ML = nullptr;
		}
		if (layer_sizes) { delete[] layer_sizes; layer_sizes = nullptr; }
		num_layers = 0;
	}

	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^ fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ readDataToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	private: System::Windows::Forms::TextBox^ textBox1;
	private: System::Windows::Forms::ToolStripMenuItem^ saveDataToolStripMenuItem;
	private: System::Windows::Forms::SaveFileDialog^ saveFileDialog1;
	private: System::Windows::Forms::ToolStripMenuItem^ processToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ trainingToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ testingToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ regressionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ mnistToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ loadMNISTToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ trainMNISTToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ testMNISTToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ trainAutoencoderToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ testReconstructionToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ trainWithEncoderToolStripMenuItem;

	private: System::Windows::Forms::DataVisualization::Charting::Chart^ chart1;
	private: System::Windows::Forms::CheckBox^ checkBoxMomentum;
	private: System::Windows::Forms::TextBox^ textBoxMomentumValue;
	private: System::Windows::Forms::Label^ labelMomentumValue;
	private: System::Windows::Forms::Button^ buttonClearCanvas;




		   /// </summary>
		   System::ComponentModel::Container^ components;

#pragma region Windows Form Designer generated code
		   /// <summary>
		   /// Erforderliche Methode f�r die Designerunterst�tzung.
		   /// Der Inhalt der Methode darf nicht mit dem Code-Editor ge�ndert werden.
		   /// </summary>
		   void InitializeComponent(void)
		   {
			   System::Windows::Forms::DataVisualization::Charting::ChartArea^ chartArea2 = (gcnew System::Windows::Forms::DataVisualization::Charting::ChartArea());
			   System::Windows::Forms::DataVisualization::Charting::Legend^ legend2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Legend());
			   System::Windows::Forms::DataVisualization::Charting::Series^ series2 = (gcnew System::Windows::Forms::DataVisualization::Charting::Series());
		   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
		   this->groupBox1 = (gcnew System::Windows::Forms::GroupBox());
			   this->labelMomentumValue = (gcnew System::Windows::Forms::Label());
			   this->textBoxMomentumValue = (gcnew System::Windows::Forms::TextBox());
			   this->checkBoxMomentum = (gcnew System::Windows::Forms::CheckBox());
			   this->Layer10TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer10 = (gcnew System::Windows::Forms::Label());
			   this->Layer9TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer9 = (gcnew System::Windows::Forms::Label());
			   this->Layer8TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer8 = (gcnew System::Windows::Forms::Label());
			   this->Layer7TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer7 = (gcnew System::Windows::Forms::Label());
			   this->Layer6TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer6 = (gcnew System::Windows::Forms::Label());
			   this->Layer5TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer5 = (gcnew System::Windows::Forms::Label());
			   this->Layer4TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer4 = (gcnew System::Windows::Forms::Label());
			   this->Layer3TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer3 = (gcnew System::Windows::Forms::Label());
			   this->Layer2TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer2 = (gcnew System::Windows::Forms::Label());
			   this->Layer1TextBox = (gcnew System::Windows::Forms::TextBox());
			   this->labelLayer1 = (gcnew System::Windows::Forms::Label());
			   this->HiddenLayerCountBox = (gcnew System::Windows::Forms::ComboBox());
			   this->labelHiddenLayers = (gcnew System::Windows::Forms::Label());
		   this->Set_Net = (gcnew System::Windows::Forms::Button());
		   this->label1 = (gcnew System::Windows::Forms::Label());
		   this->ClassCountBox = (gcnew System::Windows::Forms::ComboBox());
		   this->groupBox2 = (gcnew System::Windows::Forms::GroupBox());
			   this->label2 = (gcnew System::Windows::Forms::Label());
			   this->ClassNoBox = (gcnew System::Windows::Forms::ComboBox());
			   this->label3 = (gcnew System::Windows::Forms::Label());
			   this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			   this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->readDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->saveDataToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->processToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->trainingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->testingToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->regressionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->mnistToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->loadMNISTToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->trainMNISTToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->testMNISTToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->trainAutoencoderToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->testReconstructionToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->trainWithEncoderToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
		   this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			   this->textBox1 = (gcnew System::Windows::Forms::TextBox());
			   this->saveFileDialog1 = (gcnew System::Windows::Forms::SaveFileDialog());
			   this->chart1 = (gcnew System::Windows::Forms::DataVisualization::Charting::Chart());
			   this->buttonClearCanvas = (gcnew System::Windows::Forms::Button());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   this->groupBox1->SuspendLayout();
			   this->groupBox2->SuspendLayout();
			   this->menuStrip1->SuspendLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->BeginInit();
			   this->SuspendLayout();
			   // 
			   // pictureBox1
			   // 
			   this->pictureBox1->BackColor = System::Drawing::SystemColors::ButtonHighlight;
			   this->pictureBox1->Location = System::Drawing::Point(17, 43);
			   this->pictureBox1->Margin = System::Windows::Forms::Padding(4);
			   this->pictureBox1->Name = L"pictureBox1";
			   this->pictureBox1->Size = System::Drawing::Size(1069, 905);
			   this->pictureBox1->TabIndex = 0;
			   this->pictureBox1->TabStop = false;
			   this->pictureBox1->Paint += gcnew System::Windows::Forms::PaintEventHandler(this, &Form1::pictureBox1_Paint);
			   this->pictureBox1->MouseClick += gcnew System::Windows::Forms::MouseEventHandler(this, &Form1::pictureBox1_MouseClick);
		   // 
		   // groupBox1
		   // 
			   this->groupBox1->Controls->Add(this->labelMomentumValue);
			   this->groupBox1->Controls->Add(this->textBoxMomentumValue);
			   this->groupBox1->Controls->Add(this->checkBoxMomentum);
			   this->groupBox1->Controls->Add(this->Layer10TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer10);
			   this->groupBox1->Controls->Add(this->Layer9TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer9);
			   this->groupBox1->Controls->Add(this->Layer8TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer8);
			   this->groupBox1->Controls->Add(this->Layer7TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer7);
			   this->groupBox1->Controls->Add(this->Layer6TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer6);
			   this->groupBox1->Controls->Add(this->Layer5TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer5);
			   this->groupBox1->Controls->Add(this->Layer4TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer4);
			   this->groupBox1->Controls->Add(this->Layer3TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer3);
			   this->groupBox1->Controls->Add(this->Layer2TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer2);
			   this->groupBox1->Controls->Add(this->Layer1TextBox);
			   this->groupBox1->Controls->Add(this->labelLayer1);
			   this->groupBox1->Controls->Add(this->HiddenLayerCountBox);
		   this->groupBox1->Controls->Add(this->labelHiddenLayers);
		   this->groupBox1->Controls->Add(this->Set_Net);
		   this->groupBox1->Controls->Add(this->label1);
		   this->groupBox1->Controls->Add(this->ClassCountBox);
		   this->groupBox1->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
			   static_cast<System::Byte>(162)));
			   this->groupBox1->Location = System::Drawing::Point(1159, 62);
			   this->groupBox1->Margin = System::Windows::Forms::Padding(4);
		   this->groupBox1->Name = L"groupBox1";
			   this->groupBox1->Padding = System::Windows::Forms::Padding(4);
			   this->groupBox1->Size = System::Drawing::Size(267, 508);
		   this->groupBox1->TabIndex = 1;
		   this->groupBox1->TabStop = false;
		   this->groupBox1->Text = L"Network Architecture";
			   // 
			   // labelMomentumValue
			   // 
			   this->labelMomentumValue->AutoSize = true;
			   this->labelMomentumValue->Location = System::Drawing::Point(155, 421);
			   this->labelMomentumValue->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelMomentumValue->Name = L"labelMomentumValue";
			   this->labelMomentumValue->Size = System::Drawing::Size(54, 17);
			   this->labelMomentumValue->TabIndex = 25;
			   this->labelMomentumValue->Text = L"Value:";
			   this->labelMomentumValue->Visible = false;
			   // 
			   // textBoxMomentumValue
			   // 
			   this->textBoxMomentumValue->Location = System::Drawing::Point(207, 418);
			   this->textBoxMomentumValue->Margin = System::Windows::Forms::Padding(4);
			   this->textBoxMomentumValue->Name = L"textBoxMomentumValue";
			   this->textBoxMomentumValue->Size = System::Drawing::Size(45, 23);
			   this->textBoxMomentumValue->TabIndex = 24;
			   this->textBoxMomentumValue->Text = L"0.5";
			   this->textBoxMomentumValue->Visible = false;
			   // 
			   // checkBoxMomentum
			   // 
			   this->checkBoxMomentum->AutoSize = true;
			   this->checkBoxMomentum->Location = System::Drawing::Point(7, 420);
			   this->checkBoxMomentum->Margin = System::Windows::Forms::Padding(4);
			   this->checkBoxMomentum->Name = L"checkBoxMomentum";
			   this->checkBoxMomentum->Size = System::Drawing::Size(140, 21);
			   this->checkBoxMomentum->TabIndex = 23;
			   this->checkBoxMomentum->Text = L"Use Momentum";
			   this->checkBoxMomentum->UseVisualStyleBackColor = true;
			   this->checkBoxMomentum->Visible = false;
			   // 
			   // Layer10TextBox
			   // 
			   this->Layer10TextBox->Location = System::Drawing::Point(93, 383);
			   this->Layer10TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer10TextBox->Name = L"Layer10TextBox";
			   this->Layer10TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer10TextBox->TabIndex = 22;
			   this->Layer10TextBox->Visible = false;
			   // 
			   // labelLayer10
			   // 
			   this->labelLayer10->AutoSize = true;
			   this->labelLayer10->Location = System::Drawing::Point(9, 386);
			   this->labelLayer10->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer10->Name = L"labelLayer10";
			   this->labelLayer10->Size = System::Drawing::Size(77, 17);
			   this->labelLayer10->TabIndex = 21;
			   this->labelLayer10->Text = L"Layer 10:";
			   this->labelLayer10->Visible = false;
			   // 
			   // Layer9TextBox
			   // 
			   this->Layer9TextBox->Location = System::Drawing::Point(93, 351);
			   this->Layer9TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer9TextBox->Name = L"Layer9TextBox";
			   this->Layer9TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer9TextBox->TabIndex = 20;
			   this->Layer9TextBox->Visible = false;
			   // 
			   // labelLayer9
			   // 
			   this->labelLayer9->AutoSize = true;
			   this->labelLayer9->Location = System::Drawing::Point(9, 354);
			   this->labelLayer9->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer9->Name = L"labelLayer9";
			   this->labelLayer9->Size = System::Drawing::Size(68, 17);
			   this->labelLayer9->TabIndex = 19;
			   this->labelLayer9->Text = L"Layer 9:";
			   this->labelLayer9->Visible = false;
			   // 
			   // Layer8TextBox
			   // 
			   this->Layer8TextBox->Location = System::Drawing::Point(93, 319);
			   this->Layer8TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer8TextBox->Name = L"Layer8TextBox";
			   this->Layer8TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer8TextBox->TabIndex = 18;
			   this->Layer8TextBox->Visible = false;
			   // 
			   // labelLayer8
			   // 
			   this->labelLayer8->AutoSize = true;
			   this->labelLayer8->Location = System::Drawing::Point(9, 322);
			   this->labelLayer8->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer8->Name = L"labelLayer8";
			   this->labelLayer8->Size = System::Drawing::Size(68, 17);
			   this->labelLayer8->TabIndex = 17;
			   this->labelLayer8->Text = L"Layer 8:";
			   this->labelLayer8->Visible = false;
			   // 
			   // Layer7TextBox
			   // 
			   this->Layer7TextBox->Location = System::Drawing::Point(93, 287);
			   this->Layer7TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer7TextBox->Name = L"Layer7TextBox";
			   this->Layer7TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer7TextBox->TabIndex = 16;
			   this->Layer7TextBox->Visible = false;
			   // 
			   // labelLayer7
			   // 
			   this->labelLayer7->AutoSize = true;
			   this->labelLayer7->Location = System::Drawing::Point(9, 290);
			   this->labelLayer7->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer7->Name = L"labelLayer7";
			   this->labelLayer7->Size = System::Drawing::Size(68, 17);
			   this->labelLayer7->TabIndex = 15;
			   this->labelLayer7->Text = L"Layer 7:";
			   this->labelLayer7->Visible = false;
			   // 
			   // Layer6TextBox
			   // 
			   this->Layer6TextBox->Location = System::Drawing::Point(93, 255);
			   this->Layer6TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer6TextBox->Name = L"Layer6TextBox";
			   this->Layer6TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer6TextBox->TabIndex = 14;
			   this->Layer6TextBox->Visible = false;
			   // 
			   // labelLayer6
			   // 
			   this->labelLayer6->AutoSize = true;
			   this->labelLayer6->Location = System::Drawing::Point(9, 258);
			   this->labelLayer6->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer6->Name = L"labelLayer6";
			   this->labelLayer6->Size = System::Drawing::Size(68, 17);
			   this->labelLayer6->TabIndex = 13;
			   this->labelLayer6->Text = L"Layer 6:";
			   this->labelLayer6->Visible = false;
			   // 
			   // Layer5TextBox
			   // 
			   this->Layer5TextBox->Location = System::Drawing::Point(93, 223);
			   this->Layer5TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer5TextBox->Name = L"Layer5TextBox";
			   this->Layer5TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer5TextBox->TabIndex = 12;
			   this->Layer5TextBox->Visible = false;
			   // 
			   // labelLayer5
			   // 
			   this->labelLayer5->AutoSize = true;
			   this->labelLayer5->Location = System::Drawing::Point(9, 226);
			   this->labelLayer5->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer5->Name = L"labelLayer5";
			   this->labelLayer5->Size = System::Drawing::Size(68, 17);
			   this->labelLayer5->TabIndex = 11;
			   this->labelLayer5->Text = L"Layer 5:";
			   this->labelLayer5->Visible = false;
			   // 
			   // Layer4TextBox
			   // 
			   this->Layer4TextBox->Location = System::Drawing::Point(93, 191);
			   this->Layer4TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer4TextBox->Name = L"Layer4TextBox";
			   this->Layer4TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer4TextBox->TabIndex = 12;
			   this->Layer4TextBox->Visible = false;
			   // 
			   // labelLayer4
			   // 
			   this->labelLayer4->AutoSize = true;
			   this->labelLayer4->Location = System::Drawing::Point(9, 194);
			   this->labelLayer4->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer4->Name = L"labelLayer4";
			   this->labelLayer4->Size = System::Drawing::Size(68, 17);
			   this->labelLayer4->TabIndex = 11;
			   this->labelLayer4->Text = L"Layer 4:";
			   this->labelLayer4->Visible = false;
			   // 
			   // Layer3TextBox
			   // 
			   this->Layer3TextBox->Location = System::Drawing::Point(93, 159);
			   this->Layer3TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer3TextBox->Name = L"Layer3TextBox";
			   this->Layer3TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer3TextBox->TabIndex = 10;
			   this->Layer3TextBox->Visible = false;
			   // 
			   // labelLayer3
			   // 
			   this->labelLayer3->AutoSize = true;
			   this->labelLayer3->Location = System::Drawing::Point(9, 162);
			   this->labelLayer3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer3->Name = L"labelLayer3";
			   this->labelLayer3->Size = System::Drawing::Size(68, 17);
			   this->labelLayer3->TabIndex = 9;
			   this->labelLayer3->Text = L"Layer 3:";
			   this->labelLayer3->Visible = false;
			   // 
			   // Layer2TextBox
			   // 
			   this->Layer2TextBox->Location = System::Drawing::Point(93, 127);
			   this->Layer2TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer2TextBox->Name = L"Layer2TextBox";
			   this->Layer2TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer2TextBox->TabIndex = 8;
			   this->Layer2TextBox->Visible = false;
			   // 
			   // labelLayer2
			   // 
			   this->labelLayer2->AutoSize = true;
			   this->labelLayer2->Location = System::Drawing::Point(9, 130);
			   this->labelLayer2->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer2->Name = L"labelLayer2";
			   this->labelLayer2->Size = System::Drawing::Size(68, 17);
			   this->labelLayer2->TabIndex = 7;
			   this->labelLayer2->Text = L"Layer 2:";
			   this->labelLayer2->Visible = false;
			   // 
			   // Layer1TextBox
			   // 
			   this->Layer1TextBox->Location = System::Drawing::Point(93, 95);
			   this->Layer1TextBox->Margin = System::Windows::Forms::Padding(4);
			   this->Layer1TextBox->Name = L"Layer1TextBox";
			   this->Layer1TextBox->Size = System::Drawing::Size(159, 23);
			   this->Layer1TextBox->TabIndex = 6;
			   this->Layer1TextBox->Visible = false;
			   // 
			   // labelLayer1
			   // 
			   this->labelLayer1->AutoSize = true;
			   this->labelLayer1->Location = System::Drawing::Point(9, 98);
			   this->labelLayer1->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelLayer1->Name = L"labelLayer1";
			   this->labelLayer1->Size = System::Drawing::Size(68, 17);
			   this->labelLayer1->TabIndex = 5;
			   this->labelLayer1->Text = L"Layer 1:";
			   this->labelLayer1->Visible = false;
			   // 
			   // HiddenLayerCountBox
			   // 
			   this->HiddenLayerCountBox->FormattingEnabled = true;
			   this->HiddenLayerCountBox->Items->AddRange(gcnew cli::array< System::Object^  >(11) {
				   L"0", L"1", L"2", L"3", L"4", L"5", L"6",
					   L"7", L"8", L"9", L"10"
			   });
			   this->HiddenLayerCountBox->Location = System::Drawing::Point(147, 58);
			   this->HiddenLayerCountBox->Margin = System::Windows::Forms::Padding(4);
			   this->HiddenLayerCountBox->Name = L"HiddenLayerCountBox";
			   this->HiddenLayerCountBox->Size = System::Drawing::Size(105, 25);
			   this->HiddenLayerCountBox->TabIndex = 4;
			   this->HiddenLayerCountBox->Text = L"0";
			   this->HiddenLayerCountBox->SelectedIndexChanged += gcnew System::EventHandler(this, &Form1::HiddenLayerCountBox_SelectedIndexChanged);
			   // 
			   // labelHiddenLayers
			   // 
			   this->labelHiddenLayers->AutoSize = true;
			   this->labelHiddenLayers->Location = System::Drawing::Point(9, 62);
			   this->labelHiddenLayers->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->labelHiddenLayers->Name = L"labelHiddenLayers";
			   this->labelHiddenLayers->Size = System::Drawing::Size(118, 17);
			   this->labelHiddenLayers->TabIndex = 3;
			   this->labelHiddenLayers->Text = L"Hidden Layers:";
		   // 
		   // Set_Net
		   // 
			   this->Set_Net->Location = System::Drawing::Point(12, 450);
			   this->Set_Net->Margin = System::Windows::Forms::Padding(4);
		   this->Set_Net->Name = L"Set_Net";
			   this->Set_Net->Size = System::Drawing::Size(240, 41);
		   this->Set_Net->TabIndex = 2;
		   this->Set_Net->Text = L"Network Setting";
		   this->Set_Net->UseVisualStyleBackColor = true;
		   this->Set_Net->Click += gcnew System::EventHandler(this, &Form1::Set_Net_Click);
			   // 
			   // label1
			   // 
			   this->label1->AutoSize = true;
			   this->label1->Location = System::Drawing::Point(144, 28);
			   this->label1->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->label1->Name = L"label1";
			   this->label1->Size = System::Drawing::Size(94, 17);
			   this->label1->TabIndex = 1;
			   this->label1->Text = L"Class Count";
			   // 
			   // ClassCountBox
			   // 
			   this->ClassCountBox->FormattingEnabled = true;
			   this->ClassCountBox->Items->AddRange(gcnew cli::array< System::Object^  >(6) { L"2", L"3", L"4", L"5", L"6", L"7" });
			   this->ClassCountBox->Location = System::Drawing::Point(13, 25);
			   this->ClassCountBox->Margin = System::Windows::Forms::Padding(4);
			   this->ClassCountBox->Name = L"ClassCountBox";
			   this->ClassCountBox->Size = System::Drawing::Size(108, 25);
		   this->ClassCountBox->TabIndex = 0;
		   this->ClassCountBox->Text = L"2";
		   // 
		   // groupBox2
		   // 
		   this->groupBox2->Controls->Add(this->label2);
		   this->groupBox2->Controls->Add(this->ClassNoBox);
		   this->groupBox2->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 8.25F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
			   static_cast<System::Byte>(162)));
			   this->groupBox2->Location = System::Drawing::Point(1440, 62);
			   this->groupBox2->Margin = System::Windows::Forms::Padding(4);
		   this->groupBox2->Name = L"groupBox2";
			   this->groupBox2->Padding = System::Windows::Forms::Padding(4);
			   this->groupBox2->Size = System::Drawing::Size(253, 86);
		   this->groupBox2->TabIndex = 2;
		   this->groupBox2->TabStop = false;
		   this->groupBox2->Text = L"Data Collection";
			   // 
			   // label2
			   // 
			   this->label2->AutoSize = true;
			   this->label2->Location = System::Drawing::Point(9, 32);
			   this->label2->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
			   this->label2->Name = L"label2";
			   this->label2->Size = System::Drawing::Size(111, 17);
			   this->label2->TabIndex = 1;
			   this->label2->Text = L"Sample Label:";
			   // 
			   // ClassNoBox
			   // 
			   this->ClassNoBox->FormattingEnabled = true;
			   this->ClassNoBox->Items->AddRange(gcnew cli::array< System::Object^  >(9) {
				   L"1", L"2", L"3", L"4", L"5", L"6", L"7", L"8",
					   L"9"
			   });
			   this->ClassNoBox->Location = System::Drawing::Point(140, 28);
			   this->ClassNoBox->Margin = System::Windows::Forms::Padding(4);
			   this->ClassNoBox->Name = L"ClassNoBox";
			   this->ClassNoBox->Size = System::Drawing::Size(99, 25);
			   this->ClassNoBox->TabIndex = 0;
			   this->ClassNoBox->Text = L"1";
		   // 
		   // label3
		   // 
		   this->label3->AutoSize = true;
			   this->label3->Location = System::Drawing::Point(1436, 160);
			   this->label3->Margin = System::Windows::Forms::Padding(4, 0, 4, 0);
		   this->label3->Name = L"label3";
			   this->label3->Size = System::Drawing::Size(44, 16);
		   this->label3->TabIndex = 3;
		   this->label3->Text = L"label3";
			   // 
			   // menuStrip1
			   // 
			   this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			   this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				   this->fileToolStripMenuItem,
					   this->processToolStripMenuItem,
					   this->mnistToolStripMenuItem
			   });
			   this->menuStrip1->Location = System::Drawing::Point(0, 0);
			   this->menuStrip1->Name = L"menuStrip1";
			   this->menuStrip1->Size = System::Drawing::Size(1920, 28);
			   this->menuStrip1->TabIndex = 4;
			   this->menuStrip1->Text = L"menuStrip1";
			   // 
			   // fileToolStripMenuItem
			   // 
			   this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				   this->readDataToolStripMenuItem,
					   this->saveDataToolStripMenuItem
			   });
			   this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			   this->fileToolStripMenuItem->Size = System::Drawing::Size(46, 24);
			   this->fileToolStripMenuItem->Text = L"File";
			   // 
			   // readDataToolStripMenuItem
			   // 
			   this->readDataToolStripMenuItem->Name = L"readDataToolStripMenuItem";
			   this->readDataToolStripMenuItem->Size = System::Drawing::Size(164, 26);
			   this->readDataToolStripMenuItem->Text = L"Read_Data";
			   this->readDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::readDataToolStripMenuItem_Click);
			   // 
			   // saveDataToolStripMenuItem
			   // 
			   this->saveDataToolStripMenuItem->Name = L"saveDataToolStripMenuItem";
			   this->saveDataToolStripMenuItem->Size = System::Drawing::Size(164, 26);
			   this->saveDataToolStripMenuItem->Text = L"Save_Data";
			   this->saveDataToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::saveDataToolStripMenuItem_Click);
			   // 
			   // processToolStripMenuItem
			   // 
			   this->processToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
				   this->trainingToolStripMenuItem,
					   this->testingToolStripMenuItem, this->regressionToolStripMenuItem
			   });
			   this->processToolStripMenuItem->Name = L"processToolStripMenuItem";
			   this->processToolStripMenuItem->Size = System::Drawing::Size(72, 24);
			   this->processToolStripMenuItem->Text = L"Process";
			   // 
			   // trainingToolStripMenuItem
			   // 
			   this->trainingToolStripMenuItem->Name = L"trainingToolStripMenuItem";
			   this->trainingToolStripMenuItem->Size = System::Drawing::Size(164, 26);
			   this->trainingToolStripMenuItem->Text = L"Training";
			   this->trainingToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::trainingToolStripMenuItem_Click);
			   // 
			   // testingToolStripMenuItem
			   // 
			   this->testingToolStripMenuItem->Name = L"testingToolStripMenuItem";
			   this->testingToolStripMenuItem->Size = System::Drawing::Size(164, 26);
			   this->testingToolStripMenuItem->Text = L"Testing";
			   this->testingToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::testingToolStripMenuItem_Click);
			   // 
			   // regressionToolStripMenuItem
			   // 
			   this->regressionToolStripMenuItem->Name = L"regressionToolStripMenuItem";
			   this->regressionToolStripMenuItem->Size = System::Drawing::Size(164, 26);
			   this->regressionToolStripMenuItem->Text = L"Regression";
			   this->regressionToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::regressionToolStripMenuItem_Click);
			   // 
		   // mnistToolStripMenuItem
		   // 
		   this->mnistToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array<System::Windows::Forms::ToolStripItem^>(6) {
			   this->loadMNISTToolStripMenuItem,
			   this->trainMNISTToolStripMenuItem,
			   this->testMNISTToolStripMenuItem,
			   this->trainAutoencoderToolStripMenuItem,
			   this->testReconstructionToolStripMenuItem,
			   this->trainWithEncoderToolStripMenuItem
		   });
		   this->mnistToolStripMenuItem->Name = L"mnistToolStripMenuItem";
		   this->mnistToolStripMenuItem->Size = System::Drawing::Size(70, 24);
		   this->mnistToolStripMenuItem->Text = L"MNIST";
			   // 
			   // loadMNISTToolStripMenuItem
			   // 
			   this->loadMNISTToolStripMenuItem->Name = L"loadMNISTToolStripMenuItem";
			   this->loadMNISTToolStripMenuItem->Size = System::Drawing::Size(180, 26);
			   this->loadMNISTToolStripMenuItem->Text = L"Load Dataset";
			   this->loadMNISTToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::loadMNISTToolStripMenuItem_Click);
			   // 
			   // trainMNISTToolStripMenuItem
			   // 
			   this->trainMNISTToolStripMenuItem->Name = L"trainMNISTToolStripMenuItem";
			   this->trainMNISTToolStripMenuItem->Size = System::Drawing::Size(180, 26);
			   this->trainMNISTToolStripMenuItem->Text = L"Train MNIST";
			   this->trainMNISTToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::trainMNISTToolStripMenuItem_Click);
			   // 
		   // testMNISTToolStripMenuItem
		   // 
		   this->testMNISTToolStripMenuItem->Name = L"testMNISTToolStripMenuItem";
		   this->testMNISTToolStripMenuItem->Size = System::Drawing::Size(220, 26);
		   this->testMNISTToolStripMenuItem->Text = L"Test MNIST";
		   this->testMNISTToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::testMNISTToolStripMenuItem_Click);
		   // 
		   // trainAutoencoderToolStripMenuItem
		   // 
		   this->trainAutoencoderToolStripMenuItem->Name = L"trainAutoencoderToolStripMenuItem";
		   this->trainAutoencoderToolStripMenuItem->Size = System::Drawing::Size(220, 26);
		   this->trainAutoencoderToolStripMenuItem->Text = L"Train Autoencoder";
		   this->trainAutoencoderToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::trainAutoencoderToolStripMenuItem_Click);
		   // 
		   // testReconstructionToolStripMenuItem
		   // 
		   this->testReconstructionToolStripMenuItem->Name = L"testReconstructionToolStripMenuItem";
		   this->testReconstructionToolStripMenuItem->Size = System::Drawing::Size(230, 26);
		   this->testReconstructionToolStripMenuItem->Text = L"Test Reconstruction";
		   this->testReconstructionToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::testReconstructionToolStripMenuItem_Click);
		   // 
		   // trainWithEncoderToolStripMenuItem
		   // 
		   this->trainWithEncoderToolStripMenuItem->Name = L"trainWithEncoderToolStripMenuItem";
		   this->trainWithEncoderToolStripMenuItem->Size = System::Drawing::Size(230, 26);
		   this->trainWithEncoderToolStripMenuItem->Text = L"Train with Encoder Features";
		   this->trainWithEncoderToolStripMenuItem->Click += gcnew System::EventHandler(this, &Form1::trainWithEncoderToolStripMenuItem_Click);
		   // 
		   // openFileDialog1
			   // 
			   this->openFileDialog1->FileName = L"openFileDialog1";
			   // 
			   // textBox1
			   // 
			   this->textBox1->Location = System::Drawing::Point(1440, 192);
			   this->textBox1->Margin = System::Windows::Forms::Padding(4);
			   this->textBox1->Multiline = true;
			   this->textBox1->Name = L"textBox1";
			   this->textBox1->Size = System::Drawing::Size(467, 361);
			   this->textBox1->TabIndex = 5;
			   // 
			   // chart1
			   // 
			   chartArea2->Name = L"ChartArea1";
			   this->chart1->ChartAreas->Add(chartArea2);
			   legend2->Name = L"Legend1";
			   this->chart1->Legends->Add(legend2);
			   this->chart1->Location = System::Drawing::Point(1138, 578);
			   this->chart1->Margin = System::Windows::Forms::Padding(4);
			   this->chart1->Name = L"chart1";
			   series2->ChartArea = L"ChartArea1";
			   series2->Legend = L"Legend1";
			   series2->Name = L"Series1";
			   this->chart1->Series->Add(series2);
			   this->chart1->Size = System::Drawing::Size(769, 370);
			   this->chart1->TabIndex = 6;
			   this->chart1->Text = L"chart1";
			   // 
			   // buttonClearCanvas
			   // 
			   this->buttonClearCanvas->BackColor = System::Drawing::Color::Crimson;
			   this->buttonClearCanvas->Cursor = System::Windows::Forms::Cursors::Hand;
			   this->buttonClearCanvas->Font = (gcnew System::Drawing::Font(L"Arial", 10, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				   static_cast<System::Byte>(0)));
			   this->buttonClearCanvas->ForeColor = System::Drawing::Color::White;
			   this->buttonClearCanvas->Location = System::Drawing::Point(992, 919);
			   this->buttonClearCanvas->Name = L"buttonClearCanvas";
			   this->buttonClearCanvas->Size = System::Drawing::Size(94, 29);
			   this->buttonClearCanvas->TabIndex = 26;
			   this->buttonClearCanvas->Text = L"CLEAR";
			   this->buttonClearCanvas->UseVisualStyleBackColor = false;
			   this->buttonClearCanvas->Click += gcnew System::EventHandler(this, &Form1::buttonClearCanvas_Click);
			   // 
			   // Form1
			   // 
			   this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->ClientSize = System::Drawing::Size(1920, 1055);
			   this->Controls->Add(this->buttonClearCanvas);
			   this->Controls->Add(this->chart1);
			   this->Controls->Add(this->textBox1);
			   this->Controls->Add(this->label3);
			   this->Controls->Add(this->groupBox2);
			   this->Controls->Add(this->groupBox1);
			   this->Controls->Add(this->pictureBox1);
			   this->Controls->Add(this->menuStrip1);
			   this->MainMenuStrip = this->menuStrip1;
			   this->Margin = System::Windows::Forms::Padding(4);
			   this->Name = L"Form1";
			   this->Text = L"Form1";
			   this->WindowState = System::Windows::Forms::FormWindowState::Maximized;
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			   this->groupBox1->ResumeLayout(false);
			   this->groupBox1->PerformLayout();
			   this->groupBox2->ResumeLayout(false);
			   this->groupBox2->PerformLayout();
			   this->menuStrip1->ResumeLayout(false);
			   this->menuStrip1->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->chart1))->EndInit();
			   this->ResumeLayout(false);
			   this->PerformLayout();

		   }
		   void draw_sample(int temp_x, int temp_y, int label) {
			   Pen^ pen;// = gcnew Pen(Color::Black, 3.0f);
			   switch (label) {
			   case 0: pen = gcnew Pen(Color::Black, 3.0f); break;
			   case 1: pen = gcnew Pen(Color::Red, 3.0f); break;
			   case 2: pen = gcnew Pen(Color::Blue, 3.0f); break;
			   case 3: pen = gcnew Pen(Color::Green, 3.0f); break;
			   case 4: pen = gcnew Pen(Color::Yellow, 3.0f); break;
			   case 5: pen = gcnew Pen(Color::Orange, 3.0f); break;
			   default: pen = gcnew Pen(Color::YellowGreen, 3.0f);
			   }//switch
			   pictureBox1->CreateGraphics()->DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
			   pictureBox1->CreateGraphics()->DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
		   }//draw_sample
#pragma endregion
	private: System::Void pictureBox1_MouseClick(System::Object^ sender, System::Windows::Forms::MouseEventArgs^ e) {
		if (class_count == 0)
			MessageBox::Show("The Network Architeture should be firstly set up");
		else {
			float* x = new float[inputDim];
			int temp_x = (System::Convert::ToInt32(e->X));
			int temp_y = (System::Convert::ToInt32(e->Y));
			x[0] = float(temp_x - (pictureBox1->Width / 2));
			x[1] = float(pictureBox1->Height / 2 - temp_y);
			int label;
			int numLabel = Convert::ToInt32(ClassNoBox->Text);
			if (numLabel > class_count)
				MessageBox::Show("The class label cannot be greater than the maximum number of classes.");
			else {
				label = numLabel - 1; //D�g�ler 0 dan ba�lad���ndan, label de�eri 0 dan ba�lamas� i�in bir eksi�i al�nm��t�r
				if (numSample == 0) { //Dinamik al�nan ilk �rnek i�in sadece
					numSample = 1;
					Samples = new float[numSample * inputDim]; targets = new float[numSample];
					for (int i = 0; i < inputDim; i++)
						Samples[i] = x[i];
					targets[0] = float(label);
				}
				else {
					numSample++;
					Samples = Add_Data(Samples, numSample, x, inputDim);
					targets = Add_Labels(targets, numSample, label);
				}//else
				draw_sample(temp_x, temp_y, label);
				label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
				delete[] x;
			}//else of if (Etiket ...
		}//else
	}//pictureMouseClick
	private: System::Void pictureBox1_Paint(System::Object^ sender, System::Windows::Forms::PaintEventArgs^ e) {
		//Ana eksen do�rularini cizdir
		Pen^ pen = gcnew Pen(Color::Black, 3.0f);
		int center_width, center_height;
		center_width = (int)(pictureBox1->Width / 2);
		center_height = (int)(pictureBox1->Height / 2);
		e->Graphics->DrawLine(pen, center_width, 0, center_width, pictureBox1->Height);
		e->Graphics->DrawLine(pen, 0, center_height, pictureBox1->Width, center_height);
		
		// Draw regression curve/line if trained
		if (regression_trained && regression_mean && regression_std) {
			Pen^ regPen = gcnew Pen(Color::Blue, 2.0f);
			
			if (!regression_is_multilayer) {
				// ===== SINGLE-LAYER: Draw straight line =====
			int x1_screen = 0;
			int x2_screen = pictureBox1->Width;
			
			// Convert screen to data coordinates
			float x1_data = float(x1_screen - center_width);
			float x2_data = float(x2_screen - center_width);
			
			// Calculate y values using y = slope * x + intercept
			// Usage: Weights[0] is slope, bias[0] is intercept
			float y1_data = Weights[0] * x1_data + bias[0];
			float y2_data = Weights[0] * x2_data + bias[0];
			
			// Convert back to screen coordinates
			int y1_screen = center_height - (int)y1_data;
			int y2_screen = center_height - (int)y2_data;
			
			// Draw the regression line
			e->Graphics->DrawLine(regPen, x1_screen, y1_screen, x2_screen, y2_screen);
			}
			else {
				// ===== MULTI-LAYER: Draw curved line =====
				int step = 5;  // Sample every 5 pixels
				array<System::Drawing::Point>^ points = gcnew array<System::Drawing::Point>(pictureBox1->Width / step + 1);
				int pointCount = 0;
				
				for (int x_screen = 0; x_screen < pictureBox1->Width; x_screen += step) {
					// Convert screen to data coordinates
					float x_data = float(x_screen - center_width);
					
					// Normalize x
					float x_norm = (x_data - regression_mean[0]) / regression_std[0];
					
					// Predict y using multi-layer network
					float x_input[1] = { x_norm };
					int hidden_layer_count = num_layers - 1;
					float y_norm = Test_Forward_MultiLayer_Regression(x_input, Weights_ML, bias_ML,
						1, layer_sizes, hidden_layer_count);
					
					// Denormalize y
					float y_data = y_norm * regression_std[1] + regression_mean[1];
					
					// Convert to screen coordinates
					int y_screen = center_height - (int)y_data;
					
					// Clamp to screen bounds
					if (y_screen < 0) y_screen = 0;
					if (y_screen >= pictureBox1->Height) y_screen = pictureBox1->Height - 1;
					
					points[pointCount++] = System::Drawing::Point(x_screen, y_screen);
				}
				
				// Draw the curve
				if (pointCount > 1) {
					e->Graphics->DrawLines(regPen, points);
				}
			}
		}
	}
	private: System::Void Set_Net_Click(System::Object^ sender, System::EventArgs^ e) {
		// Network is constructed
		class_count = Convert::ToInt32(ClassCountBox->Text);
		
	// Calculate actual neuron count for output layer
		neuron_count = (class_count > 2) ? class_count : 1;
		
	// Check if multi-layer or single-layer
	int hidden_layer_count = Convert::ToInt32(HiddenLayerCountBox->Text);
	
	if (hidden_layer_count == 0) {
		// ===== SINGLE-LAYER MODE (Original Code) =====
		is_multilayer = false;
		
		// Clean up old weights and any other mode data
		Cleanup_Network();
		
		// Initialize weights for single layer
		Weights = new float[neuron_count * inputDim];
		bias = new float[neuron_count];
		Weights = init_array_random(neuron_count * inputDim);
		bias = init_array_random(neuron_count);
		
		Set_Net->Text = "Single-Layer Ready";
	}
	else {
		// ===== MULTI-LAYER MODE =====
		is_multilayer = true;
		
		// Clean up old multi-layer weights and any other mode data
		Cleanup_Network();
		
		// Layer_sizes will hold ONLY hidden layer sizes (for passing to functions)
		// Total layers for internal use = hidden + output
		num_layers = hidden_layer_count + 1;
		layer_sizes = new int[hidden_layer_count];  // Only hidden layers
		
		// Read ONLY hidden layer sizes from TextBoxes (NOT output layer)
		try {
			if (hidden_layer_count >= 1 && !String::IsNullOrWhiteSpace(Layer1TextBox->Text))
				layer_sizes[0] = Convert::ToInt32(Layer1TextBox->Text);
			else if (hidden_layer_count >= 1)
				layer_sizes[0] = 10; // Default
				
			if (hidden_layer_count >= 2 && !String::IsNullOrWhiteSpace(Layer2TextBox->Text))
				layer_sizes[1] = Convert::ToInt32(Layer2TextBox->Text);
			else if (hidden_layer_count >= 2)
				layer_sizes[1] = 8; // Default
				
			if (hidden_layer_count >= 3 && !String::IsNullOrWhiteSpace(Layer3TextBox->Text))
				layer_sizes[2] = Convert::ToInt32(Layer3TextBox->Text);
			else if (hidden_layer_count >= 3)
				layer_sizes[2] = 5; // Default
				
			if (hidden_layer_count >= 4 && !String::IsNullOrWhiteSpace(Layer4TextBox->Text))
				layer_sizes[3] = Convert::ToInt32(Layer4TextBox->Text);
			else if (hidden_layer_count >= 4)
				layer_sizes[3] = 5; // Default
				
			if (hidden_layer_count >= 5 && !String::IsNullOrWhiteSpace(Layer5TextBox->Text))
				layer_sizes[4] = Convert::ToInt32(Layer5TextBox->Text);
			else if (hidden_layer_count >= 5)
				layer_sizes[4] = 5; // Default
				
			if (hidden_layer_count >= 6 && !String::IsNullOrWhiteSpace(Layer6TextBox->Text))
				layer_sizes[5] = Convert::ToInt32(Layer6TextBox->Text);
			else if (hidden_layer_count >= 6)
				layer_sizes[5] = 5; // Default
				
			if (hidden_layer_count >= 7 && !String::IsNullOrWhiteSpace(Layer7TextBox->Text))
				layer_sizes[6] = Convert::ToInt32(Layer7TextBox->Text);
			else if (hidden_layer_count >= 7)
				layer_sizes[6] = 5; // Default
				
			if (hidden_layer_count >= 8 && !String::IsNullOrWhiteSpace(Layer8TextBox->Text))
				layer_sizes[7] = Convert::ToInt32(Layer8TextBox->Text);
			else if (hidden_layer_count >= 8)
				layer_sizes[7] = 5; // Default
				
			if (hidden_layer_count >= 9 && !String::IsNullOrWhiteSpace(Layer9TextBox->Text))
				layer_sizes[8] = Convert::ToInt32(Layer9TextBox->Text);
			else if (hidden_layer_count >= 9)
				layer_sizes[8] = 5; // Default
				
			if (hidden_layer_count >= 10 && !String::IsNullOrWhiteSpace(Layer10TextBox->Text))
				layer_sizes[9] = Convert::ToInt32(Layer10TextBox->Text);
			else if (hidden_layer_count >= 10)
				layer_sizes[9] = 5; // Default
		}
		catch (Exception^ ex) {
			MessageBox::Show("Invalid layer size! Using defaults.");
			for (int i = 0; i < hidden_layer_count; i++)
				layer_sizes[i] = 10; // Default for all
		}
		
		// Allocate weights and biases for all layers (hidden + output)
		Weights_ML = new float*[num_layers];
		bias_ML = new float*[num_layers];
		
		// Initialize hidden layers
		int input_size = inputDim;
		for (int layer = 0; layer < hidden_layer_count; layer++) {
			int output_size = layer_sizes[layer];  // Hidden layer size
			
			// Allocate weights and bias for this hidden layer
			Weights_ML[layer] = new float[output_size * input_size];
			bias_ML[layer] = new float[output_size];
			
			// Initialize with random values
			Weights_ML[layer] = init_array_random(output_size * input_size);
			bias_ML[layer] = init_array_random(output_size);
			
			// Next layer's input size is this layer's output size
			input_size = output_size;
		}
		
		// Initialize output layer (last layer)
		int output_layer_idx = num_layers - 1;
		Weights_ML[output_layer_idx] = new float[neuron_count * input_size];
		bias_ML[output_layer_idx] = new float[neuron_count];
		Weights_ML[output_layer_idx] = init_array_random(neuron_count * input_size);
		bias_ML[output_layer_idx] = init_array_random(neuron_count);
		
		// Display network architecture
		String^ arch = "Multi-Layer Ready:\n";
		arch += "Input: " + inputDim + " -> ";
		for (int i = 0; i < hidden_layer_count; i++) {
			arch += "H" + (i+1) + "(" + layer_sizes[i] + ") -> ";
		}
		arch += "Output(" + neuron_count + ")";
		
		// Print to textBox1
		textBox1->AppendText("=== Multi-Layer Network Set ===\r\n");
		textBox1->AppendText("Hidden Layers: " + hidden_layer_count + "\r\n");
		for (int i = 0; i < hidden_layer_count; i++) {
			textBox1->AppendText("  Layer " + (i+1) + ": " + layer_sizes[i] + " neurons\r\n");
		}
		textBox1->AppendText("Output Layer: " + neuron_count + " neurons\r\n");
		textBox1->AppendText("Total Weights Allocated: " + num_layers + " layers\r\n\r\n");
		
		Set_Net->Text = "Multi-Layer Ready";
		MessageBox::Show(arch);
	}
	}//Set_Net
	private: System::Void readDataToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		char** c = new char* [2];
		// Veri K�mesini okunacak 
		MessageBox::Show("Veri K�mesini Y�kleyin");
		c[0] = "../Data/Samples.txt";
		c[1] = "../Data/weights.txt";
		std::ifstream file;
		int num, w, h, Dim, label;
		file.open(c[0]);
		if (file.is_open()) {
			//MessageBox::Show("Dosya acildi");
			file >> Dim >> w >> h >> num;
			textBox1->Text += "Dimension: " + Convert::ToString(Dim) + "- Width: " + Convert::ToString(w) + " - Height: " + Convert::ToString(h) + " - Number of Class: " + Convert::ToString(num) + "\r\n";
			// Set network values
			class_count = num;
			inputDim = Dim;
			if (Weights) delete[] Weights;
			if (bias) delete[] bias;
			Weights = new float[class_count * inputDim];
			bias = new float[class_count];
			numSample = 0;
			float* x = new float[inputDim];
			while (!file.eof())
			{
				if (numSample == 0) { //ilk �rnek i�in sadece
					numSample = 1;
					Samples = new float[inputDim]; targets = new float[numSample];
					for (int i = 0; i < inputDim; i++)
						file >> Samples[i];
					file >> targets[0];
				}
				else {

					for (int i = 0; i < inputDim; i++)
						file >> x[i];
					file >> label;
					if (!file.eof()) {
						numSample++;
						Samples = Add_Data(Samples, numSample, x, inputDim);
						targets = Add_Labels(targets, numSample, label);
					}
				}//else
			} //while
			delete[]x;
			file.close();
			for (int i = 0; i < numSample; i++) {
				draw_sample(Samples[i * inputDim] + w, h - Samples[i * inputDim + 1], targets[i]);
				for (int j = 0; j < inputDim; j++)
					textBox1->Text += Convert::ToString(Samples[i * inputDim + j]) + " ";
				textBox1->Text += Convert::ToString(targets[i]) + "\r\n";
			}
			//draw_sample(temp_x, temp_y, label);
			label3->Text = "Samples Count: " + System::Convert::ToString(numSample);
			MessageBox::Show("Dosya basari ile okundu");
		}//file.is_open
		else MessageBox::Show("Dosya acilamadi");
		//Get weights
		int Layer;
		file.open(c[1]);
		if (file.is_open()) {
			file >> Layer >> Dim >> num;
			class_count = num;
			inputDim = Dim;
			if (Weights) delete[] Weights;
			if (bias) delete[] bias;
			Weights = new float[class_count * inputDim];
			bias = new float[class_count];
			textBox1->Text += "Layer: " + Convert::ToString(Layer) + " Dimension: " + Convert::ToString(Dim) + " numClass:" + Convert::ToString(num) + "\r\n";
			while (!file.eof())
			{
				for (int i = 0; i < class_count; i++)
					for (int j = 0; j < inputDim; j++)
						file >> Weights[i * inputDim + j];
				for (int i = 0; i < class_count; i++)
					file >> bias[i];
			}
			file.close();
		}//file.is_open
		delete[]c;
	}//Read_Data
	private: System::Void saveDataToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		if (numSample != 0) {
			char** c = new char* [2];
			// Veri K�mesi yaz�lacak
			c[0] = "../Data/Samples.txt";
			c[1] = "../Data/weights.txt";
			std::ofstream ofs(c[0]);
			if (!ofs.bad()) {
				// Width,  Height, number of Class, data+label
				ofs << inputDim << " " << pictureBox1->Width / 2 << " " << pictureBox1->Height / 2 << " " << class_count << std::endl;
				for (int i = 0; i < numSample; i++) {
					for (int d = 0; d < inputDim; d++)
						ofs << Samples[i * inputDim + d] << " ";
					ofs << targets[i] << std::endl;
				}
				ofs.close();
			}
			else MessageBox::Show("Samples icin dosya acilamadi");
			std::ofstream file(c[1]);
			if (!file.bad()) {
				// #Layer Dimension numClass weights biases
				file << 1 << " " << inputDim << " " << class_count << std::endl;
				for (int k = 0; k < class_count * inputDim; k++)
					file << Weights[k] << " ";
				file << std::endl;
				for (int k = 0; k < class_count; k++)
					file << bias[k] << " ";
				file.close();
			}
			else MessageBox::Show("Weight icin dosya acilamadi");
			delete[]c;
		}
		else MessageBox::Show("At least one sample should be given");
	}//Save_Data
	// Helper function to perform classification testing
	private: void PerformClassificationTest(bool showMessage) {
		// Check if training was performed
		if (!mean || !std) {
			if (showMessage) {
				MessageBox::Show("Please perform training first!");
			}
			return;
		}
		
		float* x = new float[2];
		int num, temp_x, temp_y;
		Bitmap^ surface = gcnew Bitmap(pictureBox1->Width, pictureBox1->Height);
		pictureBox1->Image = surface;
		Color c;
		for (int row = 0; row < pictureBox1->Height; row += 2) {
			for (int column = 0; column < pictureBox1->Width; column += 2) {
				x[0] = (float)(column - (pictureBox1->Width / 2));
				x[1] = (float)((pictureBox1->Height / 2) - row);
				// Use training normalization parameters
				x[0] = (x[0] - mean[0]) / std[0];
				x[1] = (x[1] - mean[1]) / std[1];
			
			// Call appropriate test function
			if (is_multilayer) {
				int hidden_layer_count = num_layers - 1;
				num = Test_Forward_MultiLayer(x, Weights_ML, bias_ML, inputDim,
											   layer_sizes, hidden_layer_count, neuron_count);
			}
			else {
				num = Test_Forward(x, Weights, bias, neuron_count, inputDim);
			}
				//MessageBox::Show("merhaba: class :" + System::Convert::ToString(numClass));
				switch (num) {
				case 0: c = Color::FromArgb(0, 0, 0); break;
				case 1: c = Color::FromArgb(255, 0, 0); break;
				case 2: c = Color::FromArgb(0, 0, 255); break; // Blue (önceden green idi)
				case 3: c = Color::FromArgb(0, 255, 0); break; // Green (önceden blue idi)
				default: c = Color::FromArgb(0, 255, 255);
				}//switch
				surface->SetPixel(column, row, c);
			}//column
		}
		
		// Draw samples on the bitmap (not on temporary graphics!)
		Graphics^ g = Graphics::FromImage(surface);
		Pen^ pen;
		if (showMessage) {
			MessageBox::Show("Örnekler çizilecek");
		}
		for (int i = 0; i < numSample; i++) {
			switch (int(targets[i])) {
			case 0: pen = gcnew Pen(Color::Black, 3.0f); break;
			case 1: pen = gcnew Pen(Color::Red, 3.0f); break;
			case 2: pen = gcnew Pen(Color::Blue, 3.0f); break;
			case 3: pen = gcnew Pen(Color::Green, 3.0f); break;
			case 4: pen = gcnew Pen(Color::Yellow, 3.0f); break;
			case 5: pen = gcnew Pen(Color::Orange, 3.0f); break;
			default: pen = gcnew Pen(Color::YellowGreen, 3.0f);
			}//switch
			temp_x = int(Samples[i * inputDim]) + pictureBox1->Width / 2;
			temp_y = pictureBox1->Height / 2 - int(Samples[i * inputDim + 1]);
			// Draw on bitmap, not temporary graphics
			g->DrawLine(pen, temp_x - 5, temp_y, temp_x + 5, temp_y);
			g->DrawLine(pen, temp_x, temp_y - 5, temp_x, temp_y + 5);
		}
		
		// Refresh the picture box to show the updated bitmap
		pictureBox1->Refresh();
		
		delete[] x;
		delete g;
	}

	private: System::Void testingToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
		PerformClassificationTest(true);
	}//Testing
private: System::Void trainingToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	int Max_epoch = 1000;
	float Min_Err = 0.001f;
	float learning_rate = 0.01f;
	int epoch = 0;

	// Clear textBox for new training info
	textBox1->Text = "";
	
	// Check if network is set up
	if (is_multilayer && (!layer_sizes || !Weights_ML)) {
		MessageBox::Show("Please set up the network first!");
		return;
	}
	if (!is_multilayer && !Weights) {
		MessageBox::Show("Please set up the network first!");
		return;
	}
	
	textBox1->Text += "=== TRAINING STARTED ===\r\n";
	textBox1->Text += "Mode: " + (is_multilayer ? "Multi-Layer" : "Single-Layer") + "\r\n";
	textBox1->Text += "Samples: " + numSample + ", Dimensions: " + inputDim + ", Classes: " + class_count + "\r\n";
	
	if (is_multilayer) {
		int hidden_layer_count = num_layers - 1;
		textBox1->Text += "Hidden Layers: " + hidden_layer_count + " | Sizes: ";
		for (int i = 0; i < hidden_layer_count; i++) {
			textBox1->Text += layer_sizes[i].ToString();
			if (i < hidden_layer_count - 1) textBox1->Text += ", ";
		}
		textBox1->Text += " | Output: " + neuron_count + "\r\n";
	}
	else {
		textBox1->Text += "Output Neurons: " + neuron_count + "\r\n";
	}
	textBox1->Text += "Learning Rate: " + learning_rate.ToString("F4") + "\r\n\r\n";

	// Calculate and save normalization parameters
	Cleanup_Normalization();
	mean = new float[inputDim];
	std = new float[inputDim];
	Z_Score_Parameters(Samples, numSample, inputDim, mean, std);
	
	// Debug normalization parameters
	String^ normMsg = "Normalization - Mean: [" + mean[0].ToString("F4") + ", " + mean[1].ToString("F4") + 
					  "] | Std: [" + std[0].ToString("F4") + ", " + std[1].ToString("F4") + "]\r\n";
	System::Diagnostics::Debug::WriteLine(normMsg);
	textBox1->Text += normMsg;

	// Normalize data for training
	float* normSamples = new float[numSample * inputDim];
	for (int i = 0; i < numSample; i++) {
		for (int j = 0; j < inputDim; j++) {
			normSamples[i * inputDim + j] = (Samples[i * inputDim + j] - mean[j]) / std[j];
		}
	}
	
	textBox1->Text += "\r\nTraining in progress...\r\n";
	
	float* error_history;
	
	if (is_multilayer) {
		// ===== MULTI-LAYER TRAINING =====
		int hidden_layer_count = num_layers - 1;
		
		// Get momentum value from checkbox and textbox
		float momentum = 0.0f;
		if (checkBoxMomentum->Checked) {
			try {
				momentum = Convert::ToSingle(textBoxMomentumValue->Text, System::Globalization::CultureInfo::InvariantCulture);
				// Clamp to valid range
				if (momentum < 0.0f) momentum = 0.0f;
				if (momentum >= 1.0f) momentum = 0.99f;
			}
			catch (...) {
				momentum = 0.5f;  // Default on error
				textBoxMomentumValue->Text = "0.5";
			}
		}
		
		textBox1->Text += "Momentum: " + (momentum > 0.0f ? momentum.ToString("F2") : "Disabled") + "\r\n";
		
		error_history = train_fcn_multilayer(normSamples, numSample, targets,
			inputDim, 
			layer_sizes,          // neuron_count: hidden layer sizes [10, 8, 5]
			hidden_layer_count,   // Layer: number of hidden layers (3)
			neuron_count,         // class_count: output layer size (e.g., 3)
			Weights_ML, bias_ML,
			learning_rate, Min_Err,
			Max_epoch, epoch,
			momentum);            // momentum: 0.0 or 0.9
	}
	else {
		// ===== SINGLE-LAYER TRAINING (Original) =====
		error_history = train_fcn(normSamples, numSample, targets,
		inputDim, neuron_count,
		Weights, bias,
		learning_rate, Min_Err,
		Max_epoch, epoch);
	}
	
	delete[] normSamples;

	if (epoch > 0) {
		// Debug output
		String^ debugMsg = "\r\n=== TRAINING COMPLETED ===\r\n";
		debugMsg += "Epochs: " + epoch + " | Final Error: " + error_history[epoch-1].ToString("F6") + "\r\n\r\n";
		
		if (is_multilayer) {
			// Multi-layer weights info
			debugMsg += "Multi-Layer Weights Summary:\r\n";
			for (int layer = 0; layer < num_layers; layer++) {
				int prev_size = (layer == 0) ? inputDim : layer_sizes[layer - 1];
				debugMsg += "  Layer " + layer + ": [" + layer_sizes[layer] + " x " + prev_size + "] = " + 
					(layer_sizes[layer] * prev_size) + " weights\r\n";
		}
		}
		else {
			// Single-layer weights details
			debugMsg += "Weights:\r\n";
		for (int i = 0; i < neuron_count; i++) {
				debugMsg += "  Neuron " + i + ": [";
				for (int j = 0; j < inputDim; j++) {
					debugMsg += Weights[i * inputDim + j].ToString("F4");
					if (j < inputDim - 1) debugMsg += ", ";
				}
				debugMsg += "]\r\n";
			}
			debugMsg += "\r\nBias: [";
			for (int i = 0; i < neuron_count; i++) {
				debugMsg += bias[i].ToString("F4");
				if (i < neuron_count - 1) debugMsg += ", ";
		}
			debugMsg += "]\r\n";
		}
		
		System::Diagnostics::Debug::WriteLine(debugMsg);
		textBox1->Text += debugMsg;
		
		chart1->Series["Series1"]->Points->Clear();
		chart1->Series["Series1"]->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
		chart1->Series["Series1"]->BorderWidth = 2;
		for (int i = 0; i < epoch; i++) {
			chart1->Series["Series1"]->Points->AddY(error_history[i]);
		}
		
		// Auto-update decision boundary if testing was previously performed
		if (pictureBox1->Image != nullptr) {
			textBox1->Text += "\r\n[Auto-updating decision boundary...]\r\n";
			PerformClassificationTest(false);  // Silent update (no MessageBox)
			textBox1->Text += "[Decision boundary updated!]\r\n";
		}
	}

	delete[] error_history;
}

private: System::Void regressionToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (numSample < 3) {
		MessageBox::Show("Please add at least 3 samples for regression!");
		return;
	}
	
	// Check if network is set up (for multi-layer regression)
	if (is_multilayer && !layer_sizes) {
		MessageBox::Show("Please set up the network first!");
		return;
	}
	
	int Max_epoch = (is_multilayer ? 5000 : 1000);  // More epochs for multi-layer
	float Min_Err = 0.001f;
	float learning_rate = (is_multilayer ? 0.001f : 0.1f);  // Much lower LR for multi-layer regression
	int epoch = 0;
	
	// Clear textBox for new regression info
	textBox1->Text = "";
	// textBox1->Text += "=== REGRESSION TRAINING STARTED ===\r\n";
	// textBox1->Text += "Mode: " + (is_multilayer ? "Multi-Layer" : "Single-Layer Linear") + "\r\n";
	// textBox1->Text += "Samples: " + numSample + ", Learning Rate: " + learning_rate.ToString("F4") + "\r\n\r\n";
	
	// Extract x and y coordinates from Samples
	float* x_data = new float[numSample];
	float* y_data = new float[numSample];
	
	for (int i = 0; i < numSample; i++) {
		x_data[i] = Samples[i * inputDim];      // x coordinate
		y_data[i] = Samples[i * inputDim + 1];  // y coordinate
	}
	
	// Calculate normalization parameters for x and y
	// Calculate normalization parameters for x and y using helper function
	float* mean_xy = new float[inputDim];
	float* std_xy = new float[inputDim];
	
	Z_Score_Parameters(Samples, numSample, inputDim, mean_xy, std_xy);
	
	// Normalize data
	float* x_norm = new float[numSample];
	float* y_norm = new float[numSample];
	
	for (int i = 0; i < numSample; i++) {
		x_norm[i] = (x_data[i] - mean_xy[0]) / std_xy[0];
		y_norm[i] = (y_data[i] - mean_xy[1]) / std_xy[1];
	}
	
	String^ normMsg = "Normalization - Mean: [" + mean_xy[0].ToString("F4") + ", " + mean_xy[1].ToString("F4") + 
					  "] | Std: [" + std_xy[0].ToString("F4") + ", " + std_xy[1].ToString("F4") + "]\r\n";
	// System::Diagnostics::Debug::WriteLine(normMsg);
	// textBox1->Text += normMsg;
	
	textBox1->Text += "\r\nTraining regression model...\r\n";
	
	float* error_history = nullptr;
	
	if (is_multilayer) {
		// ===== MULTI-LAYER REGRESSION =====
		// Prepare input data (x_norm as 1D input)
		float* x_input = new float[numSample * 1];  // 1D input
		for (int i = 0; i < numSample; i++) {
			x_input[i] = x_norm[i];
		}
		
		int hidden_layer_count = num_layers - 1;
		
		// Get momentum value from checkbox and textbox
		float momentum = 0.0f;
		if (checkBoxMomentum->Checked) {
			try {
				momentum = Convert::ToSingle(textBoxMomentumValue->Text, System::Globalization::CultureInfo::InvariantCulture);
				// Clamp to valid range
				if (momentum < 0.0f) momentum = 0.0f;
				if (momentum >= 1.0f) momentum = 0.99f;
			}
			catch (...) {
				momentum = 0.5f;  // Default on error
				textBoxMomentumValue->Text = "0.5";
			}
		}
		
		textBox1->Text += "Momentum: " + (momentum > 0.0f ? momentum.ToString("F2") : "Disabled") + "\r\n";
		
		// Train multi-layer regression
		error_history = train_fcn_multilayer_regression(x_input, numSample, y_norm,
			1,  // inputDim = 1 (only x)
			layer_sizes,          // neuron_count: hidden layer sizes
			hidden_layer_count,   // Layer: number of hidden layers
			Weights_ML, bias_ML,
			learning_rate, Min_Err, Max_epoch, epoch,
			momentum);            // momentum: 0.0 or 0.9
		
		delete[] x_input;
		
		// Mark as multi-layer regression and save normalization params
		regression_is_multilayer = true;
		regression_trained = true;
		
		// Save normalization parameters for drawing
		if (!regression_mean) regression_mean = new float[2];
		if (!regression_std) regression_std = new float[2];
		regression_mean[0] = mean_xy[0];
		regression_mean[1] = mean_xy[1];
		regression_std[0] = std_xy[0];
		regression_std[1] = std_xy[1];
		
		textBox1->Text += "\r\nMulti-layer regression trained successfully!\r\n";
		textBox1->Text += "(Curve will be drawn on canvas)\r\n";
	}
	else {
	// ===== SINGLE-LAYER LINEAR REGRESSION =====
	// Use regression_slope and regression_intercept as Weights and bias
	// We need pointers for the function, so we'll use single-element arrays concept
	// But since regression_train now expects arrays, we can just point to our member variables?
	// No, member variables are single floats. We should allocate temporary arrays 
	// or just pass address if the function treats them as arrays of size 1.
	// However, standard practice is to allocate.
	
	// Create temporary arrays for training
	float* reg_weights = new float[1];
	float* reg_bias = new float[1];
	
	// Initialize with current values (or random/zero)
	// Check if Weights exist, otherwise 0
	reg_weights[0] = (Weights != nullptr) ? Weights[0] : 0.0f;
	reg_bias[0] = (bias != nullptr) ? bias[0] : 0.0f;
	
	// Train linear regression (Single Layer: 1 input -> 1 output)
	// inputDim=1, class_count=1
	error_history = regression_train(x_norm, numSample, y_norm, 
		1, 1, 
		reg_weights, reg_bias, 
		learning_rate, Min_Err, Max_epoch, epoch, textBox1);
	
	// Copy results to class variables (Allocating if necessary)
	if (!Weights) Weights = new float[1];
	if (!bias) bias = new float[1];
	
	// Denormalize parameters directly into Weights and bias
	// y = slope*x + intercept
	Weights[0] = reg_weights[0] * (std_xy[1] / std_xy[0]);
	bias[0] = reg_bias[0] * std_xy[1] + mean_xy[1] - Weights[0] * mean_xy[0];
	
	String^ normParamsMsg = "\r\nNormalized Parameters:\r\n";
	normParamsMsg += "  Slope: " + reg_weights[0].ToString("F6") + " | Intercept: " + reg_bias[0].ToString("F6") + "\r\n";
	// System::Diagnostics::Debug::WriteLine(normParamsMsg);
	// textBox1->Text += normParamsMsg;
	
	String^ denormParamsMsg = "Denormalized Parameters:\r\n";
	denormParamsMsg += "  Slope: " + Weights[0].ToString("F6") + " | Intercept: " + bias[0].ToString("F6") + "\r\n";
	// System::Diagnostics::Debug::WriteLine(denormParamsMsg);
	// textBox1->Text += denormParamsMsg;

	// Cleanup temporary arrays
	delete[] reg_weights;
	delete[] reg_bias;
	
	// regression_slope = slope; // Removed
	// regression_intercept = intercept; // Removed
		regression_is_multilayer = false;
		regression_trained = true;
		
		// Save normalization parameters for drawing
		if (!regression_mean) regression_mean = new float[2];
		if (!regression_std) regression_std = new float[2];
		regression_mean[0] = mean_xy[0];
		regression_mean[1] = mean_xy[1];
		regression_std[0] = std_xy[0];
		regression_std[1] = std_xy[1];
	}
	
	delete[] mean_xy;
	delete[] std_xy;
	delete[] x_norm;
	delete[] y_norm;
	
	if (epoch > 0) {
		// Debug output
		// String^ resultMsg = "\r\n=== REGRESSION COMPLETED ===\r\n";
		String^ resultMsg = "\r\n\r\nEpochs: " + epoch + " | Final Error: " + error_history[epoch-1].ToString("F6") + "\r\n\r\n";
		
		/*
		if (!is_multilayer && regression_trained) {
			resultMsg += "Final Equation: y = " + Weights[0].ToString("F4") + 
			"*x + " + bias[0].ToString("F4") + "\r\n";
		}
		else if (is_multilayer) {
			resultMsg += "Multi-Layer Network trained for regression.\r\n";
			resultMsg += "Use Test menu to see predictions across the space.\r\n";
		}
		*/
		
		System::Diagnostics::Debug::WriteLine(resultMsg);
		textBox1->Text += resultMsg;
		
		// Plot error graph
		chart1->Series["Series1"]->Points->Clear();
		chart1->Series["Series1"]->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
		chart1->Series["Series1"]->BorderWidth = 2;
		for (int i = 0; i < epoch; i++) {
			chart1->Series["Series1"]->Points->AddY(error_history[i]);
		}
	}
	
	// Draw regression line on pictureBox
	pictureBox1->Refresh();
	
	delete[] x_data;
	delete[] y_data;
	delete[] error_history;
}

private: System::Void HiddenLayerCountBox_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
	int layerCount = Convert::ToInt32(HiddenLayerCountBox->Text);
	
	// Hide all layers first
	labelLayer1->Visible = false;
	Layer1TextBox->Visible = false;
	labelLayer2->Visible = false;
	Layer2TextBox->Visible = false;
	labelLayer3->Visible = false;
	Layer3TextBox->Visible = false;
	labelLayer4->Visible = false;
	Layer4TextBox->Visible = false;
	labelLayer5->Visible = false;
	Layer5TextBox->Visible = false;
	labelLayer6->Visible = false;
	Layer6TextBox->Visible = false;
	labelLayer7->Visible = false;
	Layer7TextBox->Visible = false;
	labelLayer8->Visible = false;
	Layer8TextBox->Visible = false;
	labelLayer9->Visible = false;
	Layer9TextBox->Visible = false;
	labelLayer10->Visible = false;
	Layer10TextBox->Visible = false;
	
	// Show layers based on selection
	if (layerCount >= 1) {
		labelLayer1->Visible = true;
		Layer1TextBox->Visible = true;
	}
	if (layerCount >= 2) {
		labelLayer2->Visible = true;
		Layer2TextBox->Visible = true;
	}
	if (layerCount >= 3) {
		labelLayer3->Visible = true;
		Layer3TextBox->Visible = true;
	}
	if (layerCount >= 4) {
		labelLayer4->Visible = true;
		Layer4TextBox->Visible = true;
	}
	if (layerCount >= 5) {
		labelLayer5->Visible = true;
		Layer5TextBox->Visible = true;
	}
	if (layerCount >= 6) {
		labelLayer6->Visible = true;
		Layer6TextBox->Visible = true;
	}
	if (layerCount >= 7) {
		labelLayer7->Visible = true;
		Layer7TextBox->Visible = true;
	}
	if (layerCount >= 8) {
		labelLayer8->Visible = true;
		Layer8TextBox->Visible = true;
	}
	if (layerCount >= 9) {
		labelLayer9->Visible = true;
		Layer9TextBox->Visible = true;
	}
	if (layerCount >= 10) {
		labelLayer10->Visible = true;
		Layer10TextBox->Visible = true;
	}
	
	// Show/hide momentum controls based on layer count
	if (layerCount >= 1) {
		checkBoxMomentum->Visible = true;
		textBoxMomentumValue->Visible = true;
		labelMomentumValue->Visible = true;
	}
	else {
		checkBoxMomentum->Visible = false;
		textBoxMomentumValue->Visible = false;
		labelMomentumValue->Visible = false;
	}
}
	private: System::Void checkedListBox1_SelectedIndexChanged(System::Object^ sender, System::EventArgs^ e) {
	}

private: System::Void buttonClearCanvas_Click(System::Object^ sender, System::EventArgs^ e) {
	// Stop any painting access first
	regression_trained = false;
	regression_is_multilayer = false;
	
	// Clear all samples and network data using helpers
	Cleanup_Samples();
	Cleanup_Network();
	Cleanup_Normalization();
	
	// Reset UI controls
	Set_Net->Text = "Network Setting";  // Reset button text
	
	// Clear the canvas and classification decision boundary
	pictureBox1->Image = nullptr;  // Clear classification testing bitmap
	pictureBox1->Invalidate();
	pictureBox1->Refresh();
	
	// Clear text box info
	textBox1->Clear();
	textBox1->Text = "Canvas and training data cleared! Ready for new data.\r\n";
	textBox1->Text += "Please set up the network again before training.\r\n";
	
	// Clear chart
	chart1->Series["Series1"]->Points->Clear();
	
	MessageBox::Show("Canvas and all training data cleared successfully!", "Clear Canvas", 
		MessageBoxButtons::OK, MessageBoxIcon::Information);
}

// ==================== MNIST EVENT HANDLERS ====================

private: System::Void loadMNISTToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	textBox1->Clear();
	textBox1->AppendText("=== Loading MNIST Dataset ===\r\n");
	
	try {
		// Clean up previous MNIST data
		if (mnist_train_samples) {
			delete[] mnist_train_samples;
			mnist_train_samples = nullptr;
		}
		if (mnist_train_targets) {
			delete[] mnist_train_targets;
			mnist_train_targets = nullptr;
		}
		if (mnist_test_samples) {
			delete[] mnist_test_samples;
			mnist_test_samples = nullptr;
		}
		if (mnist_test_targets) {
			delete[] mnist_test_targets;
			mnist_test_targets = nullptr;
		}
		
	// Load training dataset (100 samples per digit = 1000 total)
	textBox1->AppendText("Loading training set...\r\n");
	float* temp_train_samples = nullptr;
	float* temp_train_targets = nullptr;
	int temp_train_count = 0;
	MNISTLoader::LoadTrainDataset(mnist_base_path, 100, 
		temp_train_samples, temp_train_targets, temp_train_count);
	mnist_train_samples = temp_train_samples;
	mnist_train_targets = temp_train_targets;
	mnist_train_count = temp_train_count;
	textBox1->AppendText("Training samples loaded: " + mnist_train_count + "\r\n");
		
		// Verify training data
		if (mnist_train_samples == nullptr || mnist_train_count == 0) {
			throw gcnew Exception("Failed to load training samples!");
		}
		
	// Load test dataset (10 samples per digit = 100 total)
	textBox1->AppendText("Loading test set...\r\n");
	float* temp_test_samples = nullptr;
	float* temp_test_targets = nullptr;
	int temp_test_count = 0;
	MNISTLoader::LoadTestDataset(mnist_base_path, 10, 
		temp_test_samples, temp_test_targets, temp_test_count);
	mnist_test_samples = temp_test_samples;
	mnist_test_targets = temp_test_targets;
	mnist_test_count = temp_test_count;
	textBox1->AppendText("Test samples loaded: " + mnist_test_count + "\r\n");
		
		// Verify test data
		if (mnist_test_samples == nullptr || mnist_test_count == 0) {
			throw gcnew Exception("Failed to load test samples!");
		}
		
		// Set dataset dimensions (CRITICAL for testing!)
		mnist_input_dim = 784;  // 28x28 pixels
		mnist_class_count = 10; // 10 digits
		
		mnist_loaded = true;
		
		textBox1->AppendText("\r\n=== Dataset Summary ===\r\n");
		textBox1->AppendText("Input Size: 784 (28x28 pixels)\r\n");
		textBox1->AppendText("Classes: 10 (digits 0-9)\r\n");
		textBox1->AppendText("Training Samples: " + mnist_train_count + "\r\n");
		textBox1->AppendText("Test Samples: " + mnist_test_count + "\r\n");
		textBox1->AppendText("\r\nDataset loaded successfully!\r\n");
		
		MessageBox::Show("MNIST dataset loaded successfully!\n\nTrain: " + mnist_train_count + 
			" samples\nTest: " + mnist_test_count + " samples", 
			"Load Complete", MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Failed to load MNIST dataset!\n\nError: " + ex->Message, 
			"Load Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

private: System::Void trainMNISTToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!mnist_loaded) {
		MessageBox::Show("Please load MNIST dataset first!\n\nUse: MNIST -> Load Dataset", 
			"No Dataset", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	if (mnist_train_samples == nullptr || mnist_train_count == 0) {
		MessageBox::Show("Training data is empty!\n\nPlease load dataset again.", 
			"Data Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	
	// Show training options dialog
	TrainingDialog^ dialog = gcnew TrainingDialog();
	System::Windows::Forms::DialogResult result = dialog->ShowDialog(this);
	
	if (result != System::Windows::Forms::DialogResult::OK) {
		// User cancelled
		return;
	}
	
	// Get parameters from dialog
	float learning_rate = dialog->LearningRate;
	int Max_epoch = dialog->MaxEpochs;
	float momentum = dialog->UseMomentum ? dialog->MomentumValue : 0.0f;
	
	textBox1->Clear();
	textBox1->AppendText("=== MNIST Training ===\r\n");
	
	try {
		// Network configuration
		int mnist_input_dim = 784;  // 28x28
		int mnist_class_count = 10; // digits 0-9
		
		// Hidden layers: 128 -> 64
		int mnist_hidden_layers = 2;
		int* mnist_layer_sizes = new int[mnist_hidden_layers]; // Only hidden layers
		mnist_layer_sizes[0] = 128; // Hidden layer 1
		mnist_layer_sizes[1] = 64;  // Hidden layer 2
		// Output layer (10 neurons) will be added by train_fcn_multilayer
		
		textBox1->AppendText("Network Architecture:\r\n");
		textBox1->AppendText("  Input: 784 neurons\r\n");
		textBox1->AppendText("  Hidden 1: 128 neurons\r\n");
		textBox1->AppendText("  Hidden 2: 64 neurons\r\n");
		textBox1->AppendText("  Output: 10 neurons\r\n\r\n");
		
		// Total layers = hidden + output
		int total_layers = mnist_hidden_layers + 1;
		
		// Allocate weights and bias
		float** mnist_weights = new float*[total_layers];
		float** mnist_bias = new float*[total_layers];
		
		// Hidden layer 1: 784 inputs -> 128 neurons
		mnist_weights[0] = new float[mnist_input_dim * mnist_layer_sizes[0]];
		mnist_bias[0] = new float[mnist_layer_sizes[0]];
		
		// Hidden layer 2: 128 inputs -> 64 neurons
		mnist_weights[1] = new float[mnist_layer_sizes[0] * mnist_layer_sizes[1]];
		mnist_bias[1] = new float[mnist_layer_sizes[1]];
		
		// Output layer: 64 inputs -> 10 neurons
		mnist_weights[2] = new float[mnist_layer_sizes[1] * mnist_class_count];
		mnist_bias[2] = new float[mnist_class_count];
		
		// Initialize weights randomly
		Random^ rng = gcnew Random();
		for (int layer = 0; layer < total_layers; layer++) {
			int input_size, output_size;
			
			if (layer == 0) {
				// First hidden layer
				input_size = mnist_input_dim;
				output_size = mnist_layer_sizes[0];
			}
			else if (layer < mnist_hidden_layers) {
				// Other hidden layers
				input_size = mnist_layer_sizes[layer - 1];
				output_size = mnist_layer_sizes[layer];
			}
			else {
				// Output layer
				input_size = mnist_layer_sizes[layer - 1];
				output_size = mnist_class_count;
			}
			
			for (int i = 0; i < input_size * output_size; i++) {
				mnist_weights[layer][i] = (float)(rng->NextDouble() * 2.0 - 1.0) * 0.1f;
			}
			for (int i = 0; i < output_size; i++) {
				mnist_bias[layer][i] = 0.0f;
			}
		}
		
		// Training parameters
		float Min_Err = 0.01f;
		int epoch = 0;
		
	textBox1->AppendText("Training Parameters:\r\n");
	textBox1->AppendText("  Learning Rate: " + learning_rate + "\r\n");
	textBox1->AppendText("  Momentum: " + momentum + "\r\n");
	textBox1->AppendText("  Min Error: " + Min_Err + "\r\n");
	textBox1->AppendText("  Max Epochs: " + Max_epoch + "\r\n\r\n");
	
	// CRITICAL: Shuffle training data to prevent class order bias
	textBox1->AppendText("Shuffling training data...\r\n");
	Random^ shuffle_rng = gcnew Random();
	for (int i = mnist_train_count - 1; i > 0; i--) {
		int j = shuffle_rng->Next(0, i + 1);
		// Swap samples
		for (int k = 0; k < mnist_input_dim; k++) {
			float temp = mnist_train_samples[i * mnist_input_dim + k];
			mnist_train_samples[i * mnist_input_dim + k] = mnist_train_samples[j * mnist_input_dim + k];
			mnist_train_samples[j * mnist_input_dim + k] = temp;
		}
		// Swap targets
		for (int k = 0; k < mnist_class_count; k++) {
			float temp = mnist_train_targets[i * mnist_class_count + k];
			mnist_train_targets[i * mnist_class_count + k] = mnist_train_targets[j * mnist_class_count + k];
			mnist_train_targets[j * mnist_class_count + k] = temp;
		}
	}
	textBox1->AppendText("Data shuffled!\r\n\r\n");
	textBox1->AppendText("Training started...\r\n");
	
	// Clear chart
	chart1->Series["Series1"]->Points->Clear();
	
	// Train the network
		// Pass mnist_hidden_layers (NOT +1), because function expects ONLY hidden layer count
		float* error_history = train_fcn_multilayer(
			mnist_train_samples, mnist_train_count, mnist_train_targets,
			mnist_input_dim, mnist_layer_sizes, mnist_hidden_layers, mnist_class_count,
			mnist_weights, mnist_bias, learning_rate, Min_Err, Max_epoch, epoch, momentum
		);
		
	// Display results
	textBox1->AppendText("\r\n=== Training Complete ===\r\n");
	textBox1->AppendText("Epochs: " + epoch + "\r\n");
	textBox1->AppendText("Final Error: " + error_history[epoch - 1] + "\r\n");
	
	// Plot error history
	chart1->Series["Series1"]->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
	chart1->Series["Series1"]->BorderWidth = 2;
	for (int i = 0; i < epoch; i++) {
		chart1->Series["Series1"]->Points->AddXY(i, error_history[i]);
	}
	
	// Save weights to member variables for testing
	// First, cleanup old weights if any
	if (this->mnist_weights) {
		int old_total_layers = this->mnist_hidden_layers + 1;
		for (int i = 0; i < old_total_layers; i++) {
			if (this->mnist_weights[i]) delete[] this->mnist_weights[i];
		}
		delete[] this->mnist_weights;
	}
	if (this->mnist_bias) {
		int old_total_layers = this->mnist_hidden_layers + 1;
		for (int i = 0; i < old_total_layers; i++) {
			if (this->mnist_bias[i]) delete[] this->mnist_bias[i];
		}
		delete[] this->mnist_bias;
	}
	if (this->mnist_layer_sizes) {
		delete[] this->mnist_layer_sizes;
	}
	
	// Copy weights and parameters
	this->mnist_weights = mnist_weights;
	this->mnist_bias = mnist_bias;
	this->mnist_layer_sizes = mnist_layer_sizes;
	this->mnist_hidden_layers = mnist_hidden_layers;
	this->mnist_input_dim = mnist_input_dim;
	this->mnist_class_count = mnist_class_count;
	this->mnist_trained = true;
	
	// Cleanup
	delete[] error_history;
	// Don't delete weights/bias/layer_sizes - they're saved now!
	
	textBox1->AppendText("\r\nNetwork is ready for testing!\r\n");
	MessageBox::Show("Training completed!\n\nEpochs: " + epoch + "\n\nYou can now test the network using MNIST -> Test MNIST", 
		"Training Complete", MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Training failed!\n\nError: " + ex->Message, 
			"Training Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

private: System::Void testMNISTToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!mnist_loaded) {
		MessageBox::Show("Please load MNIST dataset first!\n\nUse: MNIST -> Load Dataset", 
			"No Dataset", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	if (!mnist_trained && !encoder_classifier_trained) {
		MessageBox::Show("Please train a network first!\n\nUse: MNIST -> Train MNIST\nor\nMNIST -> Train with Encoder Features", 
			"No Trained Network", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	textBox1->Clear();
	textBox1->AppendText("=== MNIST Testing ===\r\n\r\n");
	
	// Check which mode we're in
	bool using_encoder = encoder_classifier_trained;
	
	textBox1->AppendText("Testing Mode: ");
	textBox1->AppendText(using_encoder ? "Encoder + Classifier (64-dim features)\r\n" : "Direct Classification (784-dim input)\r\n");
	textBox1->AppendText("\r\n");
	
	// Debug: Check if weights are loaded
	if (!using_encoder) {
		textBox1->AppendText("DEBUG Info:\r\n");
		textBox1->AppendText("  mnist_trained: " + mnist_trained + "\r\n");
		textBox1->AppendText("  mnist_weights: " + (mnist_weights != nullptr ? "Not null" : "NULL!") + "\r\n");
		textBox1->AppendText("  mnist_bias: " + (mnist_bias != nullptr ? "Not null" : "NULL!") + "\r\n");
		textBox1->AppendText("  mnist_hidden_layers: " + mnist_hidden_layers + "\r\n");
		textBox1->AppendText("  mnist_input_dim: " + mnist_input_dim + "\r\n");
		textBox1->AppendText("  mnist_class_count: " + mnist_class_count + "\r\n\r\n");
	}
	
	try {
		// Test on test dataset
		int correct = 0;
		int total = mnist_test_count;
		
		// Confusion matrix: [actual_class][predicted_class]
		int confusion_matrix[10][10] = {0};
		
		textBox1->AppendText("Testing " + total + " samples...\r\n");
		if (using_encoder) {
			textBox1->AppendText("DEBUG - Encoder latent dim: " + autoencoder_latent_dim + "\r\n");
			textBox1->AppendText("DEBUG - Classifier input dim: " + autoencoder_latent_dim + "\r\n");
		}
		
		// DEBUG: Check first test target
		textBox1->AppendText("DEBUG - First test target: ");
		for (int c = 0; c < mnist_class_count; c++) {
			textBox1->AppendText(mnist_test_targets[c].ToString("F1") + " ");
		}
		textBox1->AppendText("\r\n\r\n");
		
		for (int i = 0; i < total; i++) {
			// Get current sample
			float* current_sample = &mnist_test_samples[i * mnist_input_dim];
			
			// Get actual class (from one-hot encoded target)
			int actual_class = -1;
			for (int c = 0; c < mnist_class_count; c++) {
				if (mnist_test_targets[i * mnist_class_count + c] > 0.5f) {
					actual_class = c;
					break;
				}
			}
			
			int predicted_class = -1;
			
			if (using_encoder) {
				// Use encoder + classifier
				int feature_dim = autoencoder_latent_dim;  // 10 dimensions
				
				// Input is already bipolar [-1, 1]
				float* bipolar_input = new float[mnist_input_dim];
				for (int b = 0; b < mnist_input_dim; b++) {
					bipolar_input[b] = current_sample[b];
				}
				
				// Extract features using encoder (2 layers: 128, 10)
				float** layer_outputs = new float*[3];  // input + 2 encoder layers
				layer_outputs[0] = bipolar_input;
				
				for (int layer = 0; layer < 2; layer++) {
					int prev_size = (layer == 0) ? mnist_input_dim : autoencoder_layer_sizes[layer - 1];
					int curr_size = autoencoder_layer_sizes[layer];
					
					layer_outputs[layer + 1] = new float[curr_size];
					
					for (int j = 0; j < curr_size; j++) {
						float net = encoder_bias[layer][j];
						for (int k = 0; k < prev_size; k++) {
							net += encoder_weights[layer][j * prev_size + k] * layer_outputs[layer][k];
						}
						layer_outputs[layer + 1][j] = tanh(net);
					}
				}
				
				// Classify using encoded features
				float* features = layer_outputs[2];  // 10-dim latent features
				// NO scaling - using raw encoder features!
				
				predicted_class = Test_Forward_MultiLayer(
					features,
					encoder_classifier_weights,
					encoder_classifier_bias,
					feature_dim,
					encoder_classifier_layers,
					1,  // 1 hidden layer
					mnist_class_count
				);
				
				// Cleanup
				delete[] bipolar_input;
				for (int layer = 1; layer <= 2; layer++) {
					delete[] layer_outputs[layer];
				}
				delete[] layer_outputs;
			}
			else {
				// Direct classification
				predicted_class = Test_Forward_MultiLayer(
					current_sample, 
					mnist_weights, 
					mnist_bias, 
					mnist_input_dim, 
					mnist_layer_sizes, 
					mnist_hidden_layers, 
					mnist_class_count
				);
			}
			
		// Debug: Print first 3 predictions with detailed output
		if (i < 3) {
			textBox1->AppendText("DEBUG SAMPLE " + i + ":\r\n");
			
			if (using_encoder) {
                // Show features being fed to classifier
                textBox1->AppendText("  Features (first 5): ");
                // Re-extract features just for display (we already discarded them in the loop logic above)
                // Actually, let's just use the features from the loop logic? 
                // In the loop above (line 2403), we computed features/predicted_class.
                // We don't have access to 'layer_outputs' here easily because it was deleted.
                // For debugging, let's just trust predicted_class but we really want the RAW OUTPUTS.
                
                // Re-run forward pass to get raw outputs for display
                float* features_debug = new float[autoencoder_latent_dim];
                
                // 1. Extract features (duplicate logic, simplified for debug)
                float* input_debug = new float[mnist_input_dim];
                for (int b = 0; b < mnist_input_dim; b++) input_debug[b] = current_sample[b]; // already normalized
                
                // Encoder pass
                float** enc_out = new float*[3];
                enc_out[0] = input_debug;
                for(int l=0; l<2; l++) {
                     int ps = (l==0) ? mnist_input_dim : autoencoder_layer_sizes[l-1];
                     int cs = autoencoder_layer_sizes[l];
                     enc_out[l+1] = new float[cs];
                     for(int n=0; n<cs; n++) {
                         float net = encoder_bias[l][n];
                         for(int k=0; k<ps; k++) net += encoder_weights[l][n*ps+k] * enc_out[l][k];
                         enc_out[l+1][n] = tanh(net);
                     }
                }
                for(int d=0; d<autoencoder_latent_dim; d++) features_debug[d] = enc_out[2][d];
                
                textBox1->AppendText("Feature[0]=" + features_debug[0].ToString("F3") + " ...\r\n");
                
                // Classifier pass
                float** cls_out = new float*[3]; // input, hidden, output
                cls_out[0] = features_debug;
                
                // Hidden
                int h_in = autoencoder_latent_dim;
                int h_out = encoder_classifier_layers[0];
                cls_out[1] = new float[h_out];
                for(int n=0; n<h_out; n++) {
                    float net = encoder_classifier_bias[0][n];
                    for(int k=0; k<h_in; k++) net += encoder_classifier_weights[0][n*h_in+k] * cls_out[0][k];
                    cls_out[1][n] = tanh(net);
                }
                
                // Output
                int o_in = h_out;
                int o_out = mnist_class_count;
                cls_out[2] = new float[o_out];
                textBox1->AppendText("  Classifier Raw Logits: ");
                for(int n=0; n<o_out; n++) {
                    float net = encoder_classifier_bias[1][n];
                    for(int k=0; k<o_in; k++) net += encoder_classifier_weights[1][n*o_in+k] * cls_out[1][k];
                    cls_out[2][n] = tanh(net); // Final activation
                    textBox1->AppendText(cls_out[2][n].ToString("F3") + " ");
                }
                textBox1->AppendText("\r\n");
                
                // Cleanup debug
                delete[] input_debug;
                delete[] enc_out[1]; delete[] enc_out[2]; delete[] enc_out;
                delete[] features_debug;
                delete[] cls_out[1]; delete[] cls_out[2]; delete[] cls_out;
			}
			else {
			    // Manual forward pass to get output values (only for direct classification)
			    float** layer_outputs = new float*[3];
			    layer_outputs[0] = new float[mnist_layer_sizes[0]]; // 128
			    layer_outputs[1] = new float[mnist_layer_sizes[1]]; // 64
			    layer_outputs[2] = new float[mnist_class_count];     // 10
			    
			    // Forward pass - Layer 0 (input -> hidden1)
			    for (int j = 0; j < mnist_layer_sizes[0]; j++) {
				    float net = mnist_bias[0][j];
				    for (int k = 0; k < mnist_input_dim; k++) {
					    net += mnist_weights[0][j * mnist_input_dim + k] * current_sample[k];
				    }
				    layer_outputs[0][j] = tanh(net);
			    }
			    
			    // Forward pass - Layer 1 (hidden1 -> hidden2)
			    for (int j = 0; j < mnist_layer_sizes[1]; j++) {
				    float net = mnist_bias[1][j];
				    for (int k = 0; k < mnist_layer_sizes[0]; k++) {
					    net += mnist_weights[1][j * mnist_layer_sizes[0] + k] * layer_outputs[0][k];
				    }
				    layer_outputs[1][j] = tanh(net);
			    }
			    
			    // Forward pass - Layer 2 (hidden2 -> output)
			    for (int j = 0; j < mnist_class_count; j++) {
				    float net = mnist_bias[2][j];
				    for (int k = 0; k < mnist_layer_sizes[1]; k++) {
					    net += mnist_weights[2][j * mnist_layer_sizes[1] + k] * layer_outputs[1][k];
				    }
				    layer_outputs[2][j] = tanh(net);
			    }
			    
			    textBox1->AppendText("Sample " + i + ": Actual=" + actual_class + ", Predicted=" + predicted_class + "\r\n");
			    textBox1->AppendText("  Output layer values: ");
			    for (int j = 0; j < mnist_class_count; j++) {
				    textBox1->AppendText(layer_outputs[2][j].ToString("F3") + " ");
			    }
			    textBox1->AppendText("\r\n");
			    
			    delete[] layer_outputs[0];
			    delete[] layer_outputs[1];
			    delete[] layer_outputs[2];
			    delete[] layer_outputs;
            }
		}
			
			// DEBUG: Show first 5 predictions
			if (i < 5) {
				textBox1->AppendText("Sample " + i + ": Actual=" + actual_class + ", Predicted=" + predicted_class + "\r\n");
			}
			
			// Update confusion matrix
			if (actual_class >= 0 && actual_class < 10 && predicted_class >= 0 && predicted_class < 10) {
				confusion_matrix[actual_class][predicted_class]++;
				
				if (predicted_class == actual_class) {
					correct++;
				}
			}
		}
		textBox1->AppendText("\r\n");
		
		// Calculate accuracy
		float accuracy = (float)correct / total * 100.0f;
		
		// Display results
		textBox1->AppendText("=== Test Results ===\r\n");
		textBox1->AppendText("Total Samples: " + total + "\r\n");
		textBox1->AppendText("Correct: " + correct + "\r\n");
		textBox1->AppendText("Incorrect: " + (total - correct) + "\r\n");
		textBox1->AppendText("Accuracy: " + accuracy.ToString("F2") + "%\r\n\r\n");
		
		// Display confusion matrix
		textBox1->AppendText("=== Confusion Matrix ===\r\n");
		textBox1->AppendText("(Rows: Actual, Columns: Predicted)\r\n\r\n");
		textBox1->AppendText("    ");
		for (int i = 0; i < 10; i++) {
			textBox1->AppendText(i.ToString()->PadLeft(3));
		}
		textBox1->AppendText("\r\n");
		textBox1->AppendText("   --------------------------------\r\n");
		
		for (int actual = 0; actual < 10; actual++) {
			textBox1->AppendText(actual.ToString() + " | ");
			for (int pred = 0; pred < 10; pred++) {
				textBox1->AppendText(confusion_matrix[actual][pred].ToString()->PadLeft(3));
			}
			textBox1->AppendText("\r\n");
		}
		
		// Display per-class accuracy
		textBox1->AppendText("\r\n=== Per-Class Accuracy ===\r\n");
		for (int c = 0; c < 10; c++) {
			int class_total = 0;
			int class_correct = confusion_matrix[c][c];
			for (int pred = 0; pred < 10; pred++) {
				class_total += confusion_matrix[c][pred];
			}
			if (class_total > 0) {
				float class_accuracy = (float)class_correct / class_total * 100.0f;
				textBox1->AppendText("Digit " + c + ": " + class_accuracy.ToString("F1") + "% (" + 
					class_correct + "/" + class_total + ")\r\n");
			}
		}
		
		// Show summary message
		String^ message = "Test completed!\n\n" +
			"Accuracy: " + accuracy.ToString("F2") + "%\n" +
			"Correct: " + correct + "/" + total;
		
		MessageBox::Show(message, "Test Results", 
			MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Testing failed!\n\nError: " + ex->Message, 
			"Test Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

private: System::Void trainAutoencoderToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!mnist_loaded) {
		MessageBox::Show("Please load MNIST dataset first!\n\nUse: MNIST -> Load Dataset", 
			"No Dataset", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	textBox1->Clear();
	textBox1->AppendText("=== Autoencoder Training ===\r\n\r\n");
	
	// Validate data is loaded
	if (!mnist_train_samples || mnist_train_count == 0) {
		MessageBox::Show("Training data is not loaded properly!\n\nPlease reload the dataset.", 
			"Data Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
		return;
	}
	
	try {
		// Debug info
		textBox1->AppendText("DEBUG: Checking data...\r\n");
		textBox1->AppendText("  mnist_train_samples: " + (mnist_train_samples != nullptr ? "OK" : "NULL!") + "\r\n");
		textBox1->AppendText("  mnist_train_count: " + mnist_train_count + "\r\n\r\n");
		
		// Autoencoder architecture : 784 -> 128 -> 10 -> 128 -> 784
		// Simple symmetric structure!
		int input_dim = 784;
		int output_dim = 784;  // Reconstruction output
		autoencoder_latent_dim = 10;  // Bottleneck layer (10 features!)
		
		// Define all layers EXCEPT output (train_fcn_multilayer adds output automatically)
		// Encoder: 784 -> 420 -> 10
		// Decoder: 10 -> 420
		// Output: 420 -> 784 (added by train_fcn_multilayer)
		int hidden_layers = 3;  // 2 encoder + 1 decoder (before output)
		
		// Use LOCAL variables (not member variables) for training
		int* layer_sizes = new int[hidden_layers];
		
		// Encoder layers
		layer_sizes[0] = 420;   // Encoder hidden layer (64)
		layer_sizes[1] = 10;   // Latent space (bottleneck) - 10 features!
		
		// Decoder layers (mirror of encoder, excluding final output)
		layer_sizes[2] = 420;   // Decoder hidden layer - SYMMETRIC!
		// Output layer (784) will be added by train_fcn_multilayer
		
		int total_layers = hidden_layers + 1; // Total including output
		
		// ALLOCATE weights and bias (train_fcn_multilayer does NOT allocate!)
		float** weights = new float*[total_layers];
		float** bias = new float*[total_layers];
		
		textBox1->AppendText("Allocating weights and biases...\r\n");
		
		Random^ rng = gcnew Random();
		for (int layer = 0; layer < total_layers; layer++) {
			int input_size, output_size;
			
			if (layer == 0) {
				// First layer: 784 -> 256
				input_size = input_dim;
				output_size = layer_sizes[0];
			}
			else if (layer < hidden_layers) {
				// Hidden layers
				input_size = layer_sizes[layer - 1];
				output_size = layer_sizes[layer];
			}
			else {
				// Output layer: 256 -> 784
				input_size = layer_sizes[layer - 1];
				output_size = output_dim;
			}
			
			textBox1->AppendText("  Layer " + layer + ": " + input_size + " -> " + output_size + "\r\n");
			
			weights[layer] = new float[output_size * input_size];
			bias[layer] = new float[output_size];
			
			// Standard Xavier initialization (for Tanh activation)
			float limit = sqrt(1.0f / input_size);  // Xavier: sqrt(1/n)
			for (int i = 0; i < output_size * input_size; i++) {
				weights[layer][i] = ((float)rng->NextDouble() * 2.0f - 1.0f) * limit;
			}
			for (int i = 0; i < output_size; i++) {
				bias[layer][i] = 0.0f;
			}
		}
		textBox1->AppendText("Weight allocation complete!\r\n\r\n");
		
		textBox1->AppendText("Autoencoder Architecture (Symmetric):\r\n");
		textBox1->AppendText("  Input: 784 neurons (28×28 image)\r\n");
		textBox1->AppendText("  Encoder: 784 → 420 → 10 (latent)\r\n");
		textBox1->AppendText("  Decoder: 10 (latent) → 420 → 784\r\n");
		textBox1->AppendText("  Latent Space: 10 dimensions\r\n");
		textBox1->AppendText("  Total Parameters: ~670K (wider network!)\r\n\r\n");
		
		// DON'T allocate weights/bias here - train_fcn_multilayer will do it!
		// Just declare the pointers (already nullptr from initialization)
		
		// Prepare training data: BOTH input and target must be BIPOLAR [-1,+1]!
		textBox1->AppendText("Preparing BIPOLAR training data...\r\n");
		float* autoencoder_inputs = nullptr;
		float* autoencoder_targets = nullptr;
		try {
			// Allocate bipolar input array
			autoencoder_inputs = new float[mnist_train_count * input_dim];
			autoencoder_targets = new float[mnist_train_count * output_dim];
			textBox1->AppendText("  Allocated memory for bipolar data\r\n");
			
			// Input is already normalized to [-1, 1] by MNISTLoader
			// Just copy it to the input arrays
			for (int i = 0; i < mnist_train_count * input_dim; i++) {
				float input_val = mnist_train_samples[i];
				autoencoder_inputs[i] = input_val;
				autoencoder_targets[i] = input_val;  // target = input (reconstruction!)
			}
			textBox1->AppendText("  Input & Target converted to BIPOLAR [-1,+1]\r\n\r\n");
		}
		catch (Exception^ ex) {
			textBox1->AppendText("ERROR allocating data: " + ex->Message + "\r\n");
			if (autoencoder_inputs) delete[] autoencoder_inputs;
			throw;
		}
		
		textBox1->AppendText("Training Parameters:\r\n");
		textBox1->AppendText("  Input: BIPOLAR [-1,+1] (converted from MNIST!)\r\n");
		textBox1->AppendText("  Target: BIPOLAR [-1,+1] (same as input - reconstruction!)\r\n");
		textBox1->AppendText("  Learning Rate: 0.001 (SAFE)\r\n");
		textBox1->AppendText("  Momentum: 0.0 (Stable)\r\n");
		textBox1->AppendText("  Max Epochs: 50 (faster training)\r\n");
		textBox1->AppendText("  Weight Init: Standard Xavier\r\n");
		textBox1->AppendText("  Training Samples: " + mnist_train_count + "\r\n\r\n");
		textBox1->AppendText("Calling train_fcn_multilayer...\r\n\r\n");
		
		// Train autoencoder using train_fcn_multilayer
		// It treats this as a multi-class problem where each "class" is a pixel value
		int epoch = 0;
		float* error_history = nullptr;
		
		try {
			textBox1->AppendText("Starting training loop...\r\n");
			textBox1->AppendText("  Input dim: " + input_dim + "\r\n");
			textBox1->AppendText("  Hidden layers: " + hidden_layers + "\r\n");
			textBox1->AppendText("  Output dim: " + output_dim + "\r\n");
			
			// train_fcn_multilayer - BOTH input and target are BIPOLAR [-1,+1]!
			error_history = train_fcn_multilayer(
				autoencoder_inputs,         // Input samples (BIPOLAR [-1,+1]!)
				mnist_train_count,          // Number of samples
				autoencoder_targets,        // Targets = same as input (reconstruction!)
				input_dim,                  // Input dimension (784)
				layer_sizes,                // Hidden layer sizes [420, 10, 420]
				hidden_layers,              // Number of hidden layers (3)
				output_dim,                 // Output dimension (784)
				weights,                    // Weights (allocated above)
				bias,                       // Bias (allocated above)
				0.001f,                     // learning_rate (Reduced to prevent explosion)
				0.001f,                     // min_error 
				50,                         // max_epochs 
				epoch,                      // epoch counter 
				0.0f                        // momentum (Off for stability)
			);
			
			textBox1->AppendText("Training function completed!\r\n");
		}
		catch (Exception^ ex) {
			textBox1->AppendText("ERROR in train_fcn_multilayer: " + ex->Message + "\r\n");
			if (autoencoder_inputs) delete[] autoencoder_inputs;
			if (autoencoder_targets) delete[] autoencoder_targets;
			if (layer_sizes) delete[] layer_sizes;
			throw;
		}
		
		// Get final error from error history
		float final_error = (error_history && epoch > 0) ? error_history[epoch - 1] : 0.0f;
		
		textBox1->AppendText("\r\n=== Training Complete ===\r\n");
		textBox1->AppendText("Epochs: " + epoch + "\r\n");
		textBox1->AppendText("Final Reconstruction Error: " + final_error.ToString("F6") + "\r\n\r\n");
		
		// Plot error history to chart
		textBox1->AppendText("Plotting training error to chart...\r\n");
		chart1->Series["Series1"]->Points->Clear();
		chart1->Series["Series1"]->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
		chart1->Series["Series1"]->BorderWidth = 2;
		chart1->Series["Series1"]->Color = System::Drawing::Color::Blue;
		chart1->Titles->Clear();
		chart1->Titles->Add("Autoencoder Training Error (MSE)");
		
		for (int i = 0; i < epoch; i++) {
			chart1->Series["Series1"]->Points->AddXY(i + 1, error_history[i]);
		}
		
		// Cleanup error history
		if (error_history) {
			delete[] error_history;
		}
		
		// Save trained weights to member variables
		autoencoder_num_layers = total_layers;
		autoencoder_latent_dim = 10;  // FIXED: 10-dim latent space!
		
		// Save layer sizes
		autoencoder_layer_sizes = new int[hidden_layers];
		for (int i = 0; i < hidden_layers; i++) {
			autoencoder_layer_sizes[i] = layer_sizes[i];
		}
		delete[] layer_sizes;  // Cleanup local
		
		// Save weights and bias
		autoencoder_weights = weights;  // Transfer ownership
		autoencoder_bias = bias;        // Transfer ownership
		
		textBox1->AppendText("Network saved to member variables\r\n\r\n");
		
		// Extract encoder weights (first 2 layers: 128, 10) for feature extraction
		int encoder_layer_count = 2;
		encoder_weights = new float*[encoder_layer_count];
		encoder_bias = new float*[encoder_layer_count];
		
		textBox1->AppendText("Extracting encoder (784 → 420 → 10):\r\n");
		
		for (int i = 0; i < encoder_layer_count; i++) {
			// Layer 0: 784 → 128
			// Layer 1: 128 → 10
			int curr_size = autoencoder_layer_sizes[i];
			int prev_size = (i == 0) ? input_dim : autoencoder_layer_sizes[i-1];
			int weight_count = curr_size * prev_size;
			
			textBox1->AppendText("  Layer " + i + ": " + prev_size + " → " + curr_size + "\r\n");
			
			encoder_weights[i] = new float[weight_count];
			encoder_bias[i] = new float[curr_size];
			
			// Copy from autoencoder
			for (int j = 0; j < weight_count; j++) {
				encoder_weights[i][j] = autoencoder_weights[i][j];
			}
			for (int j = 0; j < curr_size; j++) {
				encoder_bias[i][j] = autoencoder_bias[i][j];
			}
		}
		textBox1->AppendText("Encoder extracted successfully!\r\n\r\n");
		
		autoencoder_trained = true;
		delete[] autoencoder_inputs;
		delete[] autoencoder_targets;
		
		textBox1->AppendText("Encoder extracted successfully!\r\n");
		textBox1->AppendText("You can now test reconstruction or use encoder for feature extraction.\r\n");
		
		MessageBox::Show("Autoencoder training completed!\n\n" +
			"Epochs: " + epoch + "\n" +
			"Final Error: " + final_error.ToString("F6") + "\n\n" +
			"Encoder is ready for feature extraction!",
			"Training Complete", MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Autoencoder training failed!\n\nError: " + ex->Message, 
			"Training Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

private: System::Void testReconstructionToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!autoencoder_trained) {
		MessageBox::Show("Please train the autoencoder first!\n\nUse: MNIST -> Train Autoencoder", 
			"No Trained Autoencoder", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	if (!mnist_loaded) {
		MessageBox::Show("Please load MNIST dataset first!\n\nUse: MNIST -> Load Dataset", 
			"No Dataset", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	textBox1->Clear();
	textBox1->AppendText("=== Testing Autoencoder Reconstruction ===\r\n\r\n");
	
	try {
		// Test reconstruction on a few samples
		int test_samples = Math::Min(10, mnist_test_count);
		float total_error = 0.0f;
		int input_dim = 784;
		int output_dim = 784;
		
		textBox1->AppendText("Testing " + test_samples + " samples...\r\n\r\n");
		
		for (int sample = 0; sample < test_samples; sample++) {
			float* input = &mnist_test_samples[sample * input_dim];
			
			// Forward pass through autoencoder
			float** layer_outputs = new float*[autoencoder_num_layers + 1];
			layer_outputs[0] = input;  // Input layer
			
			// Forward through all layers
			for (int layer = 0; layer < autoencoder_num_layers; layer++) {
				int prev_size, curr_size;
				
				if (layer == 0) {
					prev_size = input_dim;
					curr_size = autoencoder_layer_sizes[0];
				}
				else if (layer < autoencoder_num_layers - 1) {
					prev_size = autoencoder_layer_sizes[layer - 1];
					curr_size = autoencoder_layer_sizes[layer];
				}
				else {
					// Last layer (output)
					prev_size = autoencoder_layer_sizes[layer - 1];
					curr_size = output_dim;
				}
				
				layer_outputs[layer + 1] = new float[curr_size];
				
				for (int i = 0; i < curr_size; i++) {
					float net = autoencoder_bias[layer][i];
					for (int j = 0; j < prev_size; j++) {
						net += autoencoder_weights[layer][i * prev_size + j] * layer_outputs[layer][j];
					}
					layer_outputs[layer + 1][i] = tanh(net);
				}
			}
			
			// Calculate reconstruction error (MSE)
			float sample_error = 0.0f;
			for (int i = 0; i < output_dim; i++) {
				float diff = input[i] - layer_outputs[autoencoder_num_layers][i];
				sample_error += diff * diff;
			}
			sample_error /= output_dim;
			total_error += sample_error;
			
			textBox1->AppendText("Sample " + sample + ": Reconstruction MSE = " + 
				sample_error.ToString("F6") + "\r\n");
			
			// Cleanup
			for (int layer = 1; layer <= autoencoder_num_layers; layer++) {
				delete[] layer_outputs[layer];
			}
			delete[] layer_outputs;
		}
		
		float avg_error = total_error / test_samples;
		textBox1->AppendText("\r\n=== Results ===\r\n");
		textBox1->AppendText("Average Reconstruction Error: " + avg_error.ToString("F6") + "\r\n");
		textBox1->AppendText("\r\nLower error = better reconstruction!\r\n");
		
		MessageBox::Show("Reconstruction test completed!\n\n" +
			"Average MSE: " + avg_error.ToString("F6"),
			"Test Results", MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Reconstruction test failed!\n\nError: " + ex->Message, 
			"Test Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

private: System::Void trainWithEncoderToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {
	if (!autoencoder_trained) {
		MessageBox::Show("Please train the autoencoder first!\n\nUse: MNIST -> Train Autoencoder", 
			"No Trained Autoencoder", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	if (!mnist_loaded) {
		MessageBox::Show("Please load MNIST dataset first!\n\nUse: MNIST -> Load Dataset", 
			"No Dataset", MessageBoxButtons::OK, MessageBoxIcon::Warning);
		return;
	}
	
	textBox1->Clear();
	textBox1->AppendText("=== Training Classifier with Encoder Features ===\r\n\r\n");
	
	try {
		// Step 1: Extract features using encoder
		textBox1->AppendText("Step 1: Extracting features using encoder...\r\n");
		
		int input_dim = 784;
		int feature_dim = autoencoder_latent_dim;  // 10 dimensions
		int num_classes = 10;
		
		textBox1->AppendText("  Encoder: 784 → 420 → 10\r\n");
		
		// Allocate feature arrays
		float* train_features = new float[mnist_train_count * feature_dim];
		
		// Convert targets to bipolar (-1, +1) for tanh output
		float* bipolar_targets = new float[mnist_train_count * num_classes];
		for (int i = 0; i < mnist_train_count * num_classes; i++) {
			bipolar_targets[i] = (mnist_train_targets[i] > 0.5f) ? 1.0f : -1.0f;
		}
		
		// Extract features for training data
		for (int sample = 0; sample < mnist_train_count; sample++) {
			float* original_input = &mnist_train_samples[sample * input_dim];
			float* features = &train_features[sample * feature_dim];
			
			// Input is already BIPOLAR [-1, 1]
			float* bipolar_input = new float[input_dim];
			for (int i = 0; i < input_dim; i++) {
				bipolar_input[i] = original_input[i];
			}
			
			// Forward pass through encoder (2 layers: 128, 10)
			float** layer_outputs = new float*[3];  // input + 2 hidden
			layer_outputs[0] = bipolar_input;
			
			for (int layer = 0; layer < 2; layer++) {
				int prev_size = (layer == 0) ? input_dim : autoencoder_layer_sizes[layer - 1];
				int curr_size = autoencoder_layer_sizes[layer];
				
				layer_outputs[layer + 1] = new float[curr_size];
				
				for (int i = 0; i < curr_size; i++) {
					float net = encoder_bias[layer][i];
					for (int j = 0; j < prev_size; j++) {
						net += encoder_weights[layer][i * prev_size + j] * layer_outputs[layer][j];
					}
					layer_outputs[layer + 1][i] = tanh(net);
				}
			}
			
			// Copy latent features (output of 2nd layer = 10 dims)
			// NO SCALING - autoencoder should produce varied features now!
			for (int i = 0; i < feature_dim; i++) {
				features[i] = layer_outputs[2][i];  // Direct copy, no scaling
			}
			
			// Cleanup
			delete[] bipolar_input;
			for (int layer = 1; layer <= 2; layer++) {
				delete[] layer_outputs[layer];
			}
			delete[] layer_outputs;
		}
		
		textBox1->AppendText("Features extracted: " + mnist_train_count + " samples x " + feature_dim + " features\r\n");
		
		// DEBUG: Show sample features (RAW!)
		textBox1->AppendText("DEBUG - First sample features (raw): ");
		int show_count = (feature_dim < 5) ? feature_dim : 5;
		for (int i = 0; i < show_count; i++) {
			textBox1->AppendText(train_features[i].ToString("F3") + " ");
		}
		textBox1->AppendText("...\r\n");
		textBox1->AppendText("  (Raw encoder features - no scaling!)\r\n\r\n");
		
		// Step 2: Train classifier on encoded features
		textBox1->AppendText("Step 2: Training classifier on encoded features...\r\n");
		
		// Balanced classifier: 10 -> 64 -> 10 (moderate capacity)
		int classifier_hidden_layers = 1;
		
		// Use LOCAL variables for training
		int* classifier_layer_sizes = new int[classifier_hidden_layers];
		classifier_layer_sizes[0] = 64;  // Moderate hidden layer (balance between capacity and stability)
		
		int classifier_total_layers = classifier_hidden_layers + 1;
		
		// ALLOCATE weights and bias (train_fcn_multilayer does NOT allocate!)
		float** classifier_weights = new float*[classifier_total_layers];
		float** classifier_bias = new float*[classifier_total_layers];
		
		Random^ rng = gcnew Random();
		
		int hidden_size = classifier_layer_sizes[0];  // 32
		
		// Hidden layer: 10 -> 64 (Xavier init for Tanh)
		classifier_weights[0] = new float[hidden_size * feature_dim];
		classifier_bias[0] = new float[hidden_size];
		float init_scale_hidden = sqrt(1.0f / feature_dim);  // Xavier: sqrt(1/input)
		for (int i = 0; i < hidden_size * feature_dim; i++) {
			classifier_weights[0][i] = ((float)rng->NextDouble() * 2.0f - 1.0f) * init_scale_hidden;
		}
		for (int i = 0; i < hidden_size; i++) {
			classifier_bias[0][i] = 0.0f;
		}
		
		// Output layer: 64 -> 10 (Xavier init for Tanh)
		classifier_weights[1] = new float[num_classes * hidden_size];
		classifier_bias[1] = new float[num_classes];
		float init_scale_output = sqrt(1.0f / hidden_size);  // Xavier: sqrt(1/input)
		for (int i = 0; i < num_classes * hidden_size; i++) {
			classifier_weights[1][i] = ((float)rng->NextDouble() * 2.0f - 1.0f) * init_scale_output;  // Xavier
		}
		for (int i = 0; i < num_classes; i++) {
			classifier_bias[1][i] = 0.0f;
		}
		
		textBox1->AppendText("Classifier Architecture: " + feature_dim + " -> " + hidden_size + " -> " + num_classes + "\r\n");
		textBox1->AppendText("  (Training on 10-dimensional encoded features)\r\n");
		textBox1->AppendText("Weights allocated successfully!\r\n\r\n");
		textBox1->AppendText("Training classifier...\r\n");
		textBox1->AppendText("  Architecture: 10 → 64 → 10 (balanced capacity)\r\n");
		textBox1->AppendText("  Features: RAW encoder output\r\n");
		textBox1->AppendText("  Targets: Bipolar (-1, +1) for tanh activation\r\n");
		textBox1->AppendText("  Weight Init: Xavier (sqrt(1/n))\r\n");
		textBox1->AppendText("  Learning Rate: 0.01 (moderate)\r\n");
		textBox1->AppendText("  Momentum: 0.9 (standard)\r\n");
		textBox1->AppendText("  Max Epochs: 1000 (sufficient training)\r\n");
		textBox1->AppendText("  Min Error: 0.001 (reasonable target)\r\n\r\n");
		
		// Train classifier
		int epoch = 0;
		float* error_history = train_fcn_multilayer(
			train_features,                 // Encoded features (10-dim)
			mnist_train_count,              // Number of samples
			bipolar_targets,                // BIPOLAR targets (-1, +1) for tanh!
			feature_dim,                    // Input dim = 10
			classifier_layer_sizes,         // Hidden layers [64] - balanced!
			classifier_hidden_layers,       // 1 hidden layer
			num_classes,                    // Output = 10
			classifier_weights,             // Weights (allocated above)
			classifier_bias,                // Bias (allocated above)
			0.01f,                          // learning_rate (moderate)
			0.001f,                         // min_error (reasonable)
			1000,                           // max_epochs (sufficient)
			epoch,                          // epoch counter
			0.9f                            // momentum = 0.9
		);
		
		// Save to member variables
		encoder_classifier_num_layers = classifier_hidden_layers + 1;
		encoder_classifier_layers = classifier_layer_sizes;  // Transfer ownership
		encoder_classifier_weights = classifier_weights;      // Transfer ownership
		encoder_classifier_bias = classifier_bias;            // Transfer ownership
		
		encoder_classifier_trained = true;
		
		// Cleanup train features and bipolar targets
		delete[] train_features;
		delete[] bipolar_targets;
		
		textBox1->AppendText("\r\n=== Training Complete ===\r\n");
		textBox1->AppendText("Epochs: " + epoch + "\r\n");
		if (epoch > 0 && error_history != nullptr) {
			float first_error = error_history[0];
			float last_error = error_history[epoch - 1];
			textBox1->AppendText("First Error: " + first_error.ToString("F6") + "\r\n");
			textBox1->AppendText("Final Error: " + last_error.ToString("F6") + "\r\n");
			float improvement = ((first_error - last_error) / first_error * 100.0f);
			textBox1->AppendText("Improvement: " + improvement.ToString("F2") + "%\r\n");
		}
		
		// Plot classification error to chart
		textBox1->AppendText("Plotting classification error to chart...\r\n");
		chart1->Series["Series1"]->Points->Clear();
		chart1->Series["Series1"]->ChartType = System::Windows::Forms::DataVisualization::Charting::SeriesChartType::Line;
		chart1->Series["Series1"]->BorderWidth = 2;
		chart1->Series["Series1"]->Color = System::Drawing::Color::Green;
		chart1->Titles->Clear();
		chart1->Titles->Add("Encoder-based Classifier Training Error");
		
		for (int i = 0; i < epoch; i++) {
			chart1->Series["Series1"]->Points->AddXY(i + 1, error_history[i]);
		}
		
		// Cleanup error history
		if (error_history) {
			delete[] error_history;
		}
		textBox1->AppendText("Classifier trained on " + feature_dim + "-dimensional encoded features!\r\n");
		textBox1->AppendText("You can now test using the normal 'Test MNIST' (it will auto-detect encoder usage).\r\n");
		
		MessageBox::Show("Encoder-based classifier training completed!\n\n" +
			"Epochs: " + epoch + "\n" +
			"Feature dimension: " + feature_dim + "\n\n" +
			"Use 'Test MNIST' to evaluate performance!",
			"Training Complete", MessageBoxButtons::OK, MessageBoxIcon::Information);
	}
	catch (Exception^ ex) {
		textBox1->AppendText("\r\nERROR: " + ex->Message + "\r\n");
		MessageBox::Show("Training failed!\n\nError: " + ex->Message, 
			"Training Error", MessageBoxButtons::OK, MessageBoxIcon::Error);
	}
}

	};  // End of Form1 class
}  // End of namespace CppCLRWinformsProjekt
