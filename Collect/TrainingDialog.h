#pragma once

namespace CppCLRWinformsProjekt {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Training Options Dialog for MNIST
	/// </summary>
	public ref class TrainingDialog : public System::Windows::Forms::Form
	{
	public:
		TrainingDialog(void)
		{
			InitializeComponent();
			//
			// Default values
			//
			checkBoxMomentum->Checked = false;
			textBoxMomentum->Text = "0.9";
			textBoxLearningRate->Text = "0.002";
			textBoxEpochs->Text = "100";
		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~TrainingDialog()
		{
			if (components)
			{
				delete components;
			}
		}

	public:
		// Properties to get training parameters
		property bool UseMomentum {
			bool get() { return checkBoxMomentum->Checked; }
		}

		property float MomentumValue {
			float get() {
				try {
					float value = Convert::ToSingle(textBoxMomentum->Text, 
						System::Globalization::CultureInfo::InvariantCulture);
					if (value < 0.0f) return 0.0f;
					if (value > 0.99f) return 0.99f;
					return value;
				}
				catch (...) {
					return 0.9f; // Default
				}
			}
		}

		property float LearningRate {
			float get() {
				try {
					float value = Convert::ToSingle(textBoxLearningRate->Text,
						System::Globalization::CultureInfo::InvariantCulture);
					if (value <= 0.0f) return 0.01f;
					if (value > 1.0f) return 1.0f;
					return value;
				}
				catch (...) {
					return 0.002f; // Default
				}
			}
		}

		property int MaxEpochs {
			int get() {
				try {
					int value = Convert::ToInt32(textBoxEpochs->Text);
					if (value <= 0) return 100;
					if (value > 10000) return 10000;
					return value;
				}
				catch (...) {
					return 100; // Default
				}
			}
		}

	private:
		System::Windows::Forms::GroupBox^ groupBoxMomentum;
		System::Windows::Forms::CheckBox^ checkBoxMomentum;
		System::Windows::Forms::TextBox^ textBoxMomentum;
		System::Windows::Forms::Label^ labelMomentumValue;
		System::Windows::Forms::GroupBox^ groupBoxParams;
		System::Windows::Forms::Label^ labelLearningRate;
		System::Windows::Forms::TextBox^ textBoxLearningRate;
		System::Windows::Forms::Label^ labelEpochs;
		System::Windows::Forms::TextBox^ textBoxEpochs;
		System::Windows::Forms::Button^ buttonTrain;
		System::Windows::Forms::Button^ buttonCancel;

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container ^components;

#pragma region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		void InitializeComponent(void)
		{
			this->groupBoxMomentum = (gcnew System::Windows::Forms::GroupBox());
			this->textBoxMomentum = (gcnew System::Windows::Forms::TextBox());
			this->labelMomentumValue = (gcnew System::Windows::Forms::Label());
			this->checkBoxMomentum = (gcnew System::Windows::Forms::CheckBox());
			this->groupBoxParams = (gcnew System::Windows::Forms::GroupBox());
			this->textBoxEpochs = (gcnew System::Windows::Forms::TextBox());
			this->labelEpochs = (gcnew System::Windows::Forms::Label());
			this->textBoxLearningRate = (gcnew System::Windows::Forms::TextBox());
			this->labelLearningRate = (gcnew System::Windows::Forms::Label());
			this->buttonTrain = (gcnew System::Windows::Forms::Button());
			this->buttonCancel = (gcnew System::Windows::Forms::Button());
			this->groupBoxMomentum->SuspendLayout();
			this->groupBoxParams->SuspendLayout();
			this->SuspendLayout();
			// 
			// groupBoxMomentum
			// 
			this->groupBoxMomentum->Controls->Add(this->textBoxMomentum);
			this->groupBoxMomentum->Controls->Add(this->labelMomentumValue);
			this->groupBoxMomentum->Controls->Add(this->checkBoxMomentum);
			this->groupBoxMomentum->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->groupBoxMomentum->Location = System::Drawing::Point(12, 12);
			this->groupBoxMomentum->Name = L"groupBoxMomentum";
			this->groupBoxMomentum->Size = System::Drawing::Size(360, 80);
			this->groupBoxMomentum->TabIndex = 0;
			this->groupBoxMomentum->TabStop = false;
			this->groupBoxMomentum->Text = L"Momentum Optimization";
			// 
			// textBoxMomentum
			// 
			this->textBoxMomentum->Enabled = false;
			this->textBoxMomentum->Location = System::Drawing::Point(240, 35);
			this->textBoxMomentum->Name = L"textBoxMomentum";
			this->textBoxMomentum->Size = System::Drawing::Size(100, 24);
			this->textBoxMomentum->TabIndex = 2;
			this->textBoxMomentum->Text = L"0.9";
			// 
			// labelMomentumValue
			// 
			this->labelMomentumValue->AutoSize = true;
			this->labelMomentumValue->Location = System::Drawing::Point(155, 38);
			this->labelMomentumValue->Name = L"labelMomentumValue";
			this->labelMomentumValue->Size = System::Drawing::Size(79, 18);
			this->labelMomentumValue->TabIndex = 1;
			this->labelMomentumValue->Text = L"Value (0-1):";
			// 
			// checkBoxMomentum
			// 
			this->checkBoxMomentum->AutoSize = true;
			this->checkBoxMomentum->Checked = true;
			this->checkBoxMomentum->CheckState = System::Windows::Forms::CheckState::Checked;
			this->checkBoxMomentum->Location = System::Drawing::Point(15, 37);
			this->checkBoxMomentum->Name = L"checkBoxMomentum";
			this->checkBoxMomentum->Size = System::Drawing::Size(138, 22);
			this->checkBoxMomentum->TabIndex = 0;
			this->checkBoxMomentum->Text = L"Enable Momentum";
			this->checkBoxMomentum->UseVisualStyleBackColor = true;
			this->checkBoxMomentum->CheckedChanged += gcnew System::EventHandler(this, &TrainingDialog::checkBoxMomentum_CheckedChanged);
			// 
			// groupBoxParams
			// 
			this->groupBoxParams->Controls->Add(this->textBoxEpochs);
			this->groupBoxParams->Controls->Add(this->labelEpochs);
			this->groupBoxParams->Controls->Add(this->textBoxLearningRate);
			this->groupBoxParams->Controls->Add(this->labelLearningRate);
			this->groupBoxParams->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 9, System::Drawing::FontStyle::Regular, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->groupBoxParams->Location = System::Drawing::Point(12, 98);
			this->groupBoxParams->Name = L"groupBoxParams";
			this->groupBoxParams->Size = System::Drawing::Size(360, 120);
			this->groupBoxParams->TabIndex = 1;
			this->groupBoxParams->TabStop = false;
			this->groupBoxParams->Text = L"Training Parameters";
			// 
			// textBoxEpochs
			// 
			this->textBoxEpochs->Location = System::Drawing::Point(180, 75);
			this->textBoxEpochs->Name = L"textBoxEpochs";
			this->textBoxEpochs->Size = System::Drawing::Size(160, 24);
			this->textBoxEpochs->TabIndex = 3;
			this->textBoxEpochs->Text = L"500";
			// 
			// labelEpochs
			// 
			this->labelEpochs->AutoSize = true;
			this->labelEpochs->Location = System::Drawing::Point(15, 78);
			this->labelEpochs->Name = L"labelEpochs";
			this->labelEpochs->Size = System::Drawing::Size(92, 18);
			this->labelEpochs->TabIndex = 2;
			this->labelEpochs->Text = L"Max Epochs:";
			// 
			// textBoxLearningRate
			// 
			this->textBoxLearningRate->Location = System::Drawing::Point(180, 35);
			this->textBoxLearningRate->Name = L"textBoxLearningRate";
			this->textBoxLearningRate->Size = System::Drawing::Size(160, 24);
			this->textBoxLearningRate->TabIndex = 1;
			this->textBoxLearningRate->Text = L"0.01";
			// 
			// labelLearningRate
			// 
			this->labelLearningRate->AutoSize = true;
			this->labelLearningRate->Location = System::Drawing::Point(15, 38);
			this->labelLearningRate->Name = L"labelLearningRate";
			this->labelLearningRate->Size = System::Drawing::Size(106, 18);
			this->labelLearningRate->TabIndex = 0;
			this->labelLearningRate->Text = L"Learning Rate:";
			// 
			// buttonTrain
			// 
			this->buttonTrain->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(46)), static_cast<System::Int32>(static_cast<System::Byte>(125)),
				static_cast<System::Int32>(static_cast<System::Byte>(50)));
			this->buttonTrain->Cursor = System::Windows::Forms::Cursors::Hand;
			this->buttonTrain->DialogResult = System::Windows::Forms::DialogResult::OK;
			this->buttonTrain->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->buttonTrain->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10.2F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->buttonTrain->ForeColor = System::Drawing::Color::White;
			this->buttonTrain->Location = System::Drawing::Point(62, 235);
			this->buttonTrain->Name = L"buttonTrain";
			this->buttonTrain->Size = System::Drawing::Size(120, 40);
			this->buttonTrain->TabIndex = 2;
			this->buttonTrain->Text = L"TRAIN";
			this->buttonTrain->UseVisualStyleBackColor = false;
			// 
			// buttonCancel
			// 
			this->buttonCancel->BackColor = System::Drawing::Color::FromArgb(static_cast<System::Int32>(static_cast<System::Byte>(244)), static_cast<System::Int32>(static_cast<System::Byte>(67)),
				static_cast<System::Int32>(static_cast<System::Byte>(54)));
			this->buttonCancel->Cursor = System::Windows::Forms::Cursors::Hand;
			this->buttonCancel->DialogResult = System::Windows::Forms::DialogResult::Cancel;
			this->buttonCancel->FlatStyle = System::Windows::Forms::FlatStyle::Flat;
			this->buttonCancel->Font = (gcnew System::Drawing::Font(L"Microsoft Sans Serif", 10.2F, System::Drawing::FontStyle::Bold, System::Drawing::GraphicsUnit::Point,
				static_cast<System::Byte>(0)));
			this->buttonCancel->ForeColor = System::Drawing::Color::White;
			this->buttonCancel->Location = System::Drawing::Point(202, 235);
			this->buttonCancel->Name = L"buttonCancel";
			this->buttonCancel->Size = System::Drawing::Size(120, 40);
			this->buttonCancel->TabIndex = 3;
			this->buttonCancel->Text = L"CANCEL";
			this->buttonCancel->UseVisualStyleBackColor = false;
			// 
			// TrainingDialog
			// 
			this->AcceptButton = this->buttonTrain;
			this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			this->CancelButton = this->buttonCancel;
			this->ClientSize = System::Drawing::Size(384, 291);
			this->Controls->Add(this->buttonCancel);
			this->Controls->Add(this->buttonTrain);
			this->Controls->Add(this->groupBoxParams);
			this->Controls->Add(this->groupBoxMomentum);
			this->FormBorderStyle = System::Windows::Forms::FormBorderStyle::FixedDialog;
			this->MaximizeBox = false;
			this->MinimizeBox = false;
			this->Name = L"TrainingDialog";
			this->StartPosition = System::Windows::Forms::FormStartPosition::CenterParent;
			this->Text = L"MNIST Training Options";
			this->groupBoxMomentum->ResumeLayout(false);
			this->groupBoxMomentum->PerformLayout();
			this->groupBoxParams->ResumeLayout(false);
			this->groupBoxParams->PerformLayout();
			this->ResumeLayout(false);

		}
#pragma endregion

	private: System::Void checkBoxMomentum_CheckedChanged(System::Object^ sender, System::EventArgs^ e) {
		textBoxMomentum->Enabled = checkBoxMomentum->Checked;
		labelMomentumValue->Enabled = checkBoxMomentum->Checked;
	}
	};
}




