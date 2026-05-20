# 🧠 Brain Tumor Detection Using Deep Learning (VGG16)

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A deep learning project that detects brain tumors from MRI images using **Transfer Learning with VGG16**. The model classifies MRI scans into **4 categories** and is deployed as a **Flask web application** with a professional dark-themed UI.

---

## 📸 Demo

> Upload any MRI image → Get instant tumor classification with confidence score

| No Tumor | Glioma | Meningioma | Pituitary |
|---|---|---|---|
| ✅ Normal scan | ⚠️ Malignant | ⚡ Usually benign | 🔬 Near pituitary gland |

---

## 📁 Project Structure

```
BrainTumorDetection/
│
├── brain_tumor_colab.ipynb        ← Full training notebook (run on Google Colab)
├── brain_tumor_interface.html     ← Standalone web interface
├── README.md                      ← You are here
│
└── flask_app/
    ├── app.py                     ← Flask backend
    ├── brain_tumor_model.h5       ← Trained model (download separately)
    └── templates/
        └── index.html             ← Flask HTML template
```

---

## 📊 Dataset

**Dataset used:** Brain Tumor MRI Dataset by Masoud Nickparvar

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

| Class | Training Images | Testing Images |
|---|---|---|
| Glioma | 1,321 | 300 |
| Meningioma | 1,339 | 306 |
| No Tumor | 1,595 | 405 |
| Pituitary | 1,457 | 300 |
| **Total** | **5,712** | **1,311** |

---

## 🏗️ Model Architecture

```
Input (128×128×3)
        ↓
   VGG16 Base (pretrained on ImageNet — frozen)
        ↓
   GlobalAveragePooling2D
        ↓
   BatchNormalization
        ↓
   Dense(512) + ReLU + Dropout(0.5)      ← Hidden Layer 1
        ↓
   Dense(256) + ReLU + Dropout(0.4)      ← Hidden Layer 2
        ↓
   Dense(128) + ReLU + Dropout(0.3)      ← Hidden Layer 3
        ↓
   Dense(4) + Softmax                    ← Output Layer
```

### Why VGG16?
- Pretrained on **14 million ImageNet images**
- Extracts powerful feature representations out of the box
- Much higher accuracy than training a CNN from scratch
- Reduces training time significantly

---

## ⚙️ How It Works

### 1. Data Preprocessing
- Images resized to **128×128 pixels**
- Pixel values normalized using **StandardScaler**
- Labels encoded using **LabelEncoder**
- Dataset split: **80% train / 20% validation**

### 2. Data Augmentation
To prevent overfitting, these augmentations are applied during training:
- Random rotation (±20°)
- Horizontal flip
- Brightness adjustment (0.8–1.2)
- Zoom (±20%)

### 3. Transfer Learning (2-Phase Training)
- **Phase 1:** VGG16 base frozen → only top layers train
- **Phase 2:** Last 4 VGG16 layers unfrozen → fine-tuning with lower learning rate (0.00001)

### 4. Training Configuration
| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Sparse Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 10 (EarlyStopping applied) |

### 5. Anti-Overfitting Techniques
- Dropout layers (0.5, 0.4, 0.3)
- L2 Regularization (0.001)
- Data Augmentation
- EarlyStopping (patience=3)
- ReduceLROnPlateau (factor=0.5)

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~90–95% |
| Loss | < 0.3 |

Evaluation includes:
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix
- Accuracy & Loss curves

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open `brain_tumor_colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: `Runtime → Change Runtime Type → T4 GPU`
3. Upload `archive.zip` dataset to Google Drive
4. Run all cells top to bottom

### Option 2 — Flask Web App (Local)

**Step 1 — Install dependencies**
```bash
pip install flask flask-cors tensorflow pillow numpy
```

**Step 2 — Place your trained model**
```
flask_app/
└── brain_tumor_model.h5   ← put your trained model here
```

**Step 3 — Run Flask**
```bash
cd flask_app
python app.py
```

**Step 4 — Open the interface**

Open `brain_tumor_interface.html` in your browser — the app is ready!

---

## 🛠️ Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| TensorFlow / Keras | Model building & training |
| VGG16 | Transfer learning backbone |
| OpenCV | Image loading & preprocessing |
| Scikit-learn | LabelEncoder, StandardScaler, metrics |
| Flask | Web backend |
| Flask-CORS | Cross-origin requests |
| HTML / CSS / JS | Frontend interface |
| Google Colab | GPU training environment |
| Matplotlib / Seaborn | Visualization |

---

## 📋 Requirements

```
tensorflow
flask
flask-cors
pillow
numpy
scikit-learn
matplotlib
seaborn
opencv-python
```

Install all:
```bash
pip install tensorflow flask flask-cors pillow numpy scikit-learn matplotlib seaborn opencv-python
```

---

## 👤 Author

**Syed Istaqbal Mehdi Gillani**
Reg. No: 23108368
BS Artificial Intelligence — 5B
SZABIST Islamabad
Department of Robotics & Artificial Intelligence

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a qualified medical professional for any health concerns.
