# Brain Tumor Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-VGG16-red?style=flat-square&logo=keras)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

A deep learning project that detects brain tumors from MRI images using Transfer Learning with VGG16. The model classifies MRI scans into four categories and is deployed as a Flask web application with a professional dark-themed interface.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [How It Works](#how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [Disclaimer](#disclaimer)

---

## Overview

Brain tumors are one of the most serious and life-threatening medical conditions. Early and accurate detection is critical for effective treatment. Traditional diagnosis relies on manual analysis of MRI scans by radiologists, which is time-consuming and prone to human error.

This project automates the detection process using a Convolutional Neural Network based on the VGG16 architecture pretrained on ImageNet. The model is trained to classify brain MRI images into four categories with high accuracy.

---

## Dataset

**Dataset:** Brain Tumor MRI Dataset by Masoud Nickparvar

**Download:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

| Class | Training Images | Testing Images |
|---|---|---|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

The dataset contains MRI brain scan images in JPG and PNG format, divided into Training and Testing folders with four class subfolders each.

---

## Model Architecture

```
Input Layer        (128 x 128 x 3)
       |
VGG16 Base         Pretrained on ImageNet — all layers frozen
       |
GlobalAveragePooling2D
       |
BatchNormalization
       |
Dense(512) + ReLU + Dropout(0.5)      Hidden Layer 1
       |
Dense(256) + ReLU + Dropout(0.4)      Hidden Layer 2
       |
Dense(128) + ReLU + Dropout(0.3)      Hidden Layer 3
       |
Dense(4)   + Softmax                  Output Layer
```

### Why VGG16

VGG16 is a deep convolutional neural network pretrained on over 14 million ImageNet images. By using transfer learning, the model starts with powerful feature extraction capabilities already built in. This results in significantly higher accuracy compared to training a CNN from scratch, especially with a limited medical dataset.

---

## How It Works

### Step 1 — Data Preprocessing

- Images resized to 128 x 128 pixels using OpenCV
- Pixel values normalized using StandardScaler (zero mean, unit variance)
- Labels encoded using LabelEncoder
- Dataset split: 80% training / 20% validation

### Step 2 — Data Augmentation

Augmentation is applied during training to prevent overfitting and improve generalization:

| Technique | Value |
|---|---|
| Rotation | +/- 20 degrees |
| Horizontal Flip | Enabled |
| Brightness | 0.8 to 1.2 |
| Zoom | +/- 20% |
| Width Shift | 15% |
| Height Shift | 15% |

### Step 3 — Transfer Learning (Two Phase Training)

**Phase 1 — Feature Learning**
The VGG16 base is fully frozen. Only the newly added dense layers are trained. The model learns to classify tumor types using the features extracted by VGG16.

**Phase 2 — Fine Tuning**
The last 4 layers of VGG16 are unfrozen. The entire model is retrained at a much lower learning rate (0.00001) to fine-tune the pretrained weights for MRI-specific features.

### Step 4 — Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Phase 1 Learning Rate | 0.0001 |
| Phase 2 Learning Rate | 0.00001 |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 32 |
| Max Epochs | 10 |
| Early Stopping Patience | 3 |

### Step 5 — Anti-Overfitting Techniques

| Technique | Purpose |
|---|---|
| Dropout (0.5, 0.4, 0.3) | Randomly disables neurons during training |
| L2 Regularization (0.001) | Penalizes large weights |
| Data Augmentation | Increases training variety |
| EarlyStopping | Stops when validation loss stops improving |
| ReduceLROnPlateau | Reduces learning rate when model plateaus |

---

## Results

| Metric | Score |
|---|---|
| Train Accuracy | 96.92% |
| Validation Accuracy | 95.80% |
| Test Accuracy | 89.44% |
| Test Loss | 0.966 |

Evaluation outputs include:
- Classification Report (Precision, Recall, F1-Score per class)
- Confusion Matrix
- Training and Validation Accuracy curves
- Training and Validation Loss curves

---

## Project Structure

```
Brain-Tumor-Detection/
|
|-- brain_tumor_colab.ipynb       Full training notebook (Google Colab)
|-- brain_tumor_interface.html    Standalone web interface
|-- requirements.txt              Python dependencies
|-- README.md                     Project documentation
|
|-- flask_app/
    |-- app.py                    Flask backend server
    |-- brain_tumor_model.h5      Trained model (train and add manually)
    |-- templates/
        |-- index.html            Flask HTML template
```

---

## How to Run

### Option 1 — Google Colab (Recommended for Training)

1. Open `brain_tumor_colab.ipynb` in Google Colab
2. Set runtime to T4 GPU: Runtime > Change Runtime Type > T4 GPU
3. Upload dataset zip to Google Drive
4. Run all cells top to bottom
5. Download the saved `brain_tumor_model.h5` from Google Drive

### Option 2 — Flask Web App (Local Deployment)

**Install dependencies**

```bash
pip install flask flask-cors tensorflow pillow numpy scikit-learn
```

**Place the trained model**

```
flask_app/
|-- app.py
|-- brain_tumor_model.h5    <- place your trained model here
```

**Run Flask**

```bash
cd flask_app
python app.py
```

**Open the interface**

Open `brain_tumor_interface.html` in your browser. The app connects to Flask at http://127.0.0.1:5000 automatically.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.8+ | Core programming language |
| TensorFlow / Keras | Model building and training |
| VGG16 | Transfer learning backbone |
| OpenCV | Image loading and preprocessing |
| Scikit-learn | LabelEncoder, StandardScaler, metrics |
| Flask | Web backend server |
| Flask-CORS | Cross-origin request handling |
| HTML / CSS / JavaScript | Frontend web interface |
| Google Colab | GPU training environment |
| Matplotlib / Seaborn | Training visualization |

---

## Author

**Name:** Syed Istaqbal Mehdi Gillani   
**Program:** BS Artificial Intelligence
**University:** SZABIST Islamabad  
**Department:** Robotics and Artificial Intelligence  

---

## Disclaimer

This project is for educational purposes only. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns related to brain tumors or any other medical condition.

---

## License

This project is licensed under the MIT License.
