# CIFAR-10 Image Classification with Deep CNNs

This project implements a custom deep convolutional neural network (CNN) architecture for image classification on the CIFAR-10 dataset, achieving a test accuracy of **92.42%**. The model uses adaptive multi-path convolutional blocks with learned feature weighting and regularization techniques.

---

## Project Overview

- Designed a deep CNN from scratch using **PyTorch**
- Constructed six adaptive intermediate blocks with **learned weighted outputs**
- Applied **Batch Normalization**, **Dropout**, **Data Augmentation**, and **Cosine Learning Rate Scheduling**
- Achieved **92.42% test accuracy** on CIFAR-10 using SGD and CrossEntropyLoss

---

## Architecture Highlights

- 6 intermediate blocks, each with multiple parallel convolutional layers
- Learned weights for each convolutional path using a fully connected layer and global average pooling
- Progressive channel expansion: 3→32→64→96→128→192→256
- Regularized using dropout (0.2–0.3), BatchNorm, and ReLU activations
- Final layers: Global Avg Pool → FC → FC → 10-class output

---

## Training Details

- **Optimizer:** SGD with momentum (0.9)  
- **Learning Rate:** 0.1 with CosineAnnealingLR (`T_max = 150`)  
- **Epochs:** 150  
- **Batch Size:** 64  
- **Loss:** CrossEntropyLoss  
- **Augmentation:** Random Crop, Flip, Color Jitter, Normalization  

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/cifar10-deep-cnn.git
cd cifar10-deep-cnn
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open and execute notebooks/cifar10_training.ipynb to train or evaluate the model.

---

See report.pdf for a detailed breakdown of the architecture, techniques, and results.
