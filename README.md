# EEG Emotion Recognition using LSTM and GRU

A deep learning project for classifying emotions from EEG signal features using recurrent neural networks.

The project compares two TensorFlow/Keras models:

1. LSTM
2. GRU

The goal is to classify EEG samples into three emotion classes:

- Negative
- Neutral
- Positive

---

## Project Overview

This project uses EEG signal features to recognize human emotional states.  
The dataset contains statistical and frequency-domain features extracted from EEG recordings.

The workflow includes:

- Data loading and inspection
- Missing value and duplicate checking
- Label encoding
- Feature scaling
- Reshaping flat features into temporal EEG sequences
- Training LSTM and GRU models
- Evaluating models using accuracy, classification reports, and confusion matrices

---

## Dataset

The dataset contains:

| Item | Value |
|---|---:|
| Samples | 2,132 |
| Feature columns | 2,548 |
| Classes | 3 |
| EEG channels | 14 |

The emotion classes are:

```text
NEGATIVE
NEUTRAL
POSITIVE
```

The 2,548 feature columns are reshaped into:

```text
182 timesteps × 14 channels = 2,548 features
```

Final model input shape:

```text
(batch_size, 182, 14)
```

---

## Preprocessing

The preprocessing pipeline includes:

- Checking for missing values
- Checking for duplicate samples
- Encoding emotion labels
- Applying StandardScaler normalization
- Splitting the dataset into training, validation, and testing sets
- Reshaping features for recurrent neural networks

Data split:

| Split | Ratio |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

---

## Models

## 1. LSTM Model

The LSTM model is designed to capture temporal dependencies in EEG signals.

Main structure:

```text
Input Layer
LSTM(64, return_sequences=True)
Dropout
LSTM(32)
Dropout
Dense
Output Layer
```

The output layer uses Softmax activation for 3-class classification.

---

## 2. GRU Model

The GRU model has a similar structure to the LSTM model but uses GRU layers.

Main structure:

```text
Input Layer
GRU(64, return_sequences=True)
Dropout
GRU(32)
Dropout
Dense
Output Layer
```

GRU is generally more lightweight than LSTM and can train faster because it uses fewer gates.

---

## Training Setup

| Setting | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Batch size | 32 |
| Epochs | 20 |
| Gradient clipping | clipnorm = 1.0 |
| Loss function | Sparse Categorical Crossentropy |
| Early stopping | Enabled |

---

## Results

| Model | Test Accuracy | Macro F1-score | Weighted Precision |
|---|---:|---:|---:|
| LSTM | 91.87% | 0.92 | 0.92 |
| GRU | **93.12%** | **0.93** | **0.93** |

The GRU model achieved the best overall test accuracy.

---

## Confusion Matrix Notes

- LSTM misclassified more Negative samples as Positive.
- GRU showed better sensitivity for the Negative class.
- Both models performed strongly on the Neutral class.

---

## Repository Structure

Recommended structure:

```text
EEG-Emotion-Recognition/
│
├── README.md
├── EEG_Emotion_Recognition_LSTM_GRU_dola.ipynb
│
├── models/
│   ├── lstm_model.keras
│   └── gru_model.keras
│
├── results/
│   ├── lstm_accuracy_loss.png
│   ├── gru_accuracy_loss.png
│   ├── lstm_confusion_matrix.png
│   └── gru_confusion_matrix.png
│
└── data/
    └── README.md
```

Note: The dataset file is not included in this repository if it is too large or private.

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EEG-Emotion-Recognition.git
cd EEG-Emotion-Recognition
```

2. Install dependencies:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

3. Open the notebook:

```text
EEG_Emotion_Recognition_LSTM_GRU_dola.ipynb
```

4. Run the notebook cells in order.

---

## Requirements

```text
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

## Conclusion

This project shows that recurrent neural networks can effectively classify emotional states from EEG signal features.

Both LSTM and GRU achieved high performance, with GRU achieving the best test accuracy of 93.12%.

---

## Notes

- This project is for learning and experimentation.
- The model is not intended for medical or clinical diagnosis.
- Results may vary depending on preprocessing, random seed, and train-test split.
