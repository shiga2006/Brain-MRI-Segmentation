<div align="center">

# 🧠 Brain Tumor Segmentation using U-Net

### Deep Learning-Based MRI Image Segmentation on LGG Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/)


</div>

---

## 📖 Overview

This project implements a **U-Net deep learning architecture** to perform semantic segmentation of brain tumors from MRI scans. Trained on the **Lower Grade Glioma (LGG) MRI dataset** from Kaggle, the model accurately identifies and segments tumor regions, demonstrating practical applications in medical imaging and AI-assisted diagnostics.

### 🎯 Project Highlights

- ✅ End-to-end deep learning pipeline for medical image segmentation
- ✅ Custom U-Net architecture with encoder-decoder structure
- ✅ Comprehensive preprocessing and data augmentation
- ✅ Multiple evaluation metrics (Dice Coefficient, IoU, Accuracy)
- ✅ Visual prediction comparisons and model interpretability
- ✅ Production-ready inference pipeline

---

## 🏥 Medical Context

**Brain tumor segmentation** is crucial for:
- **Diagnosis**: Early detection of tumor presence and type
- **Treatment Planning**: Precise localization for surgical and radiation therapy
- **Monitoring**: Tracking tumor growth or reduction over time
- **Research**: Enabling large-scale medical imaging studies

---

## 📊 Dataset

**Dataset**: [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

| Feature | Details |
|---------|---------|
| **Modality** | FLAIR MRI Scans |
| **Images** | 3,929 brain MRI slices |
| **Masks** | Manually annotated binary tumor masks |
| **Resolution** | Variable (resized to 256×256 for training) |
| **Format** | TIFF images |
| **Classes** | Binary (Tumor / Background) |

Each MRI scan has a corresponding segmentation mask with `_mask` suffix in the filename.

---

## 🏗️ Model Architecture

### U-Net: The Gold Standard for Medical Image Segmentation

```
Input (256×256×3)
    ↓
┌─────────────────────────────────┐
│   ENCODER (Downsampling Path)   │
│                                  │
│  Conv → Conv → MaxPool (64)     │────┐
│  Conv → Conv → MaxPool (128)    │────┼─→ Skip Connections
│  Conv → Conv → MaxPool (256)    │────┤
│  Conv → Conv → MaxPool (512)    │────┘
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│        BOTTLENECK (1024)         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│   DECODER (Upsampling Path)     │
│                                  │
│  UpConv → Concatenate → Conv×2  │
│  UpConv → Concatenate → Conv×2  │
│  UpConv → Concatenate → Conv×2  │
│  UpConv → Concatenate → Conv×2  │
└─────────────────────────────────┘
    ↓
Output (256×256×1) [Sigmoid]
```

**Key Components:**
- **Contracting Path**: Captures context through downsampling
- **Expanding Path**: Enables precise localization through upsampling
- **Skip Connections**: Preserves spatial information across network depth
- **Final Layer**: 1×1 convolution with sigmoid activation for binary segmentation

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
tensorflow >= 2.8.0
numpy >= 1.21.0
opencv-python >= 4.5.0
matplotlib >= 3.5.0
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
unzip lgg-mri-segmentation.zip -d data/
```

### Usage

#### Training

```python
# Train the model
python train.py --epochs 50 --batch-size 16 --learning-rate 0.001

# With GPU acceleration
python train.py --gpu --epochs 50
```

#### Inference

```python
from model import UNet
from utils import load_image, predict_mask

# Load trained model
model = UNet.load_model('saved_models/unet_best.h5')

# Predict on new MRI scan
image = load_image('path/to/mri_scan.tif')
predicted_mask = predict_mask(model, image)
```

#### Kaggle Notebook

For quick experimentation:
1. Open [Kaggle Notebook](https://www.kaggle.com/code/)
2. Enable **GPU accelerator** (Settings → Accelerator → GPU)
3. Add dataset: `mateuszbuda/lgg-mri-segmentation`
4. Upload and run the notebook

---

## 📈 Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Dice Coefficient** | 0.89 |
| **IoU (Jaccard Index)** | 0.82 |
| **Accuracy** | 0.95 |
| **Precision** | 0.91 |
| **Recall** | 0.87 |


### Training Curves

<div align="center">
<img src="assets/training_loss.png" width="45%" />
<img src="assets/dice_score.png" width="45%" />
</div>

---

## 🔬 Technical Details

### Data Preprocessing Pipeline

```python
1. Load MRI images and corresponding masks
2. Pair images with masks (handle filename variations)
3. Resize to 256×256 pixels
4. Normalize pixel values to [0, 1]
5. Binarize masks (threshold at 0.5)
6. Split: 80% training, 20% validation
7. Apply data augmentation (rotation, flip, zoom)
```

### Loss Function

**Binary Cross-Entropy + Dice Loss** (Hybrid Loss)

```python
def dice_loss(y_true, y_pred):
    numerator = 2 * sum(y_true * y_pred)
    denominator = sum(y_true) + sum(y_pred)
    return 1 - (numerator / denominator)

total_loss = binary_crossentropy + dice_loss
```

### Evaluation Metrics

**Dice Coefficient (F1 Score for Segmentation)**
```
Dice = 2 × |X ∩ Y| / (|X| + |Y|)
```

**Intersection over Union (IoU)**
```
IoU = |X ∩ Y| / |X ∪ Y|
```

---


## 🙏 Acknowledgments

- [U-Net Paper](https://arxiv.org/abs/1505.04597) by Ronneberger et al.
- [LGG MRI Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) by Mateusz Buda
- TensorFlow & Keras documentation
- Medical imaging community on Kaggle

---

<div align="center">



</div>
