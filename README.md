# Multi-Category Text Classification using Deep Learning Models

This repository contains the implementation of two advanced neural network architectures for a multi-category text classification task. The project explores and analyzes the performance of a hybrid CNN-LSTM model and a Transformer-based model.

---

## 📝 Project Overview
The primary goal of this project is to design, implement, and evaluate deep learning models for classifying text documents into one of five categories. Two distinct approaches were investigated:

**Part A: A Hybrid CNN-LSTM Architecture**  
This model (`PART_A.py`) combines Convolutional Neural Networks (CNNs) for local feature extraction and Long Short-Term Memory (LSTM) networks to capture sequential dependencies. The architecture is further enhanced with a self-attention mechanism and dynamic meta-embeddings.

**Part B: A Transformer-based Text Encoder**  
This model (`PART_B.py`) leverages the self-attention mechanism of the Transformer architecture to build a powerful text classifier, analyzing its performance with respect to architectural depth and the inclusion of positional embeddings.

The performance of all models is evaluated using the **micro-average F1 score** on the test set.

---

## 📂 Dataset
The project utilizes a dataset designed for multi-category text classification. The data is split into separate training and testing files (`TrainData.csv` and `TestLabels.csv`). Place them inside the `Datasets/` folder.

Categories:
- business  
- tech  
- politics  
- sport  
- entertainment  

---

## 🤖 Models & Architectures

### Part A: CNN-LSTM with Self-Attention
Implemented in `PART_A.py` as a highly modular architecture (`ModularTextClassifier`) that allows for easy experimentation by enabling or disabling its core components.

**Core Components:**
- **Embedding Layer:** Converts input token IDs into dense vectors.  
- **CNN Branch (`use_cnn`):** Captures local n-gram features using multiple kernel sizes.  
- **LSTM Branch (`use_lstm`):** Models long-range dependencies in the sequence.  
- **Self-Attention (`use_attention`):** Weighs token importance in the LSTM output.  
- **Meta-Embedding (`use_embedding`):** Uses averaged embeddings as an alternative feature path.  
- **Fusion Layer:** Concatenates outputs from active branches and sends them to the final classifier.  

**Key Findings (Part A):**  
FastText embeddings + self-attention achieved **micro-average F1 score of 0.9714**.

---

### Part B: Transformer-based Classifier
Implemented in `PART_B.py`, this model is a custom encoder-only Transformer built from scratch in PyTorch.

**Core Components & Techniques:**
- **Advanced Preprocessing:** Regex-based tokenizer for granular text splitting.  
- **Learnable Positional Encoding:** Trains positional encodings instead of using sinusoidal ones.  
- **Custom Transformer Encoder:** Stack of `CustomTransformerBlock` with multi-head attention and feed-forward layers.  
- **Label Smoothing:** Improves generalization with `LabelSmoothingCrossEntropy`.  
- **Data Augmentation:** Random deletion of words during training.  
- **Global Context Pooling:** Uses mean pooling across encoder outputs.  

**Key Findings (Part B):**  
The best result came from **1 Encoder Block with no positional encoding**, achieving **micro-average F1 score of 0.9469**.

---

## 🏗️ Code Structure
```
.
├── Datasets/
│   ├── TrainData.csv
│   ├── TestLabels.csv
├── LICENSE
├── PART_A.py          # CNN-LSTM model
├── PART_B.py          # Transformer model
```

---

## ⚙️ Setup and Usage

Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install dependencies:
```
torch
pandas
numpy
scikit-learn
tqdm
```

```bash
pip install -r requirements.txt
```

Ensure the dataset files are placed inside the `Datasets/` folder.

Run models:

**CNN-LSTM (Part A):**
```bash
python PART_A.py
```

**Transformer (Part B):**
```bash
python PART_B.py
```

---

## 🔬 Running Experiments

### Part A Experiments
Edit `PART_A.py` to toggle architectural components:
```python
use_cnn = True
use_lstm = True
use_embedding = False
use_attention = False
```

### Part B Experiments
Edit `PART_B.py` to configure encoder layers and positional encoding:
```python
num_encoder_layers = 4
use_pos = True
```

---

## 📊 Results Summary

| Model Architecture | Best Configuration | Micro-average F1 Score |
|---------------------|--------------------|-------------------------|
| **CNN-LSTM**        | FastText Embeddings + Self-Attention | **0.9714** |
| **Transformer**     | 1 Encoder Block, No Positional Encoding | 0.9469 |

**Conclusion:**  
The CNN-LSTM model with pre-trained embeddings slightly outperformed the Transformer-based model. Hybrid models remain competitive for text classification tasks.

---
