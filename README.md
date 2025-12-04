# EmotionVision-CNN  
Deep learning system for multi-class emotion recognition from facial images, built with PyTorch, Inception-V3, and ResNet-18.

---

## Overview
This project develops and evaluates deep learning models for **image-based sentiment classification**, predicting one of six universal emotions:

- Happy  
- Sad  
- Anger  
- Pain  
- Fear  
- Disgust  

The system explores two major CNN architectures:
- **Inception-V3 with transfer learning**
- **ResNet-18 with and without data augmentation**

It also includes a full machine-learning workflow covering data preprocessing, augmentation, stratified splitting, training, evaluation, fairness across classes, and overfitting diagnostics.

---

## Objectives
- Build a sentiment-recognition model capable of handling subtle and overlapping emotions.
- Study class imbalance and gradient bias.
- Evaluate performance with accuracy, macro-F1, per-class recall, ROC-AUC, and confusion matrices.
- Compare baseline vs. augmented models under the same training budget.
- Conduct overfitting tests, including a **label-shuffling sanity check**.

---

## Key Features
### **✔ Complete Training Pipeline**
- Custom dataset loader  
- Inception-V3 + auxiliary classifier  
- ResNet-18 baseline  
- Weighted cross-entropy for class imbalance  
- AdamW optimizer + LR scheduler  
- Early stopping logic

### **✔ Data Augmentation Module**
- Horizontal flip  
- Small random rotations  
- Color jitter  
- Gaussian blur  
- Random resized crop  

### **✔ Evaluation Tools**
- Classification report (precision, recall, F1)  
- Confusion matrix  
- Per-class ROC-AUC  
- Macro-F1 tracking  
- Learning curve visualization  
- Overfitting analysis & sanity checks

---

## Project Structure

EmotionVision-CNN/
│
├── data/                     # Emotion class folders (anger, happy, etc.)
├── src/
│   ├── dataset.py            # Custom dataset class
│   ├── transforms.py         # Preprocessing & augmentation
│   ├── train.py              # Training loops
│   ├── evaluate.py           # Evaluation utilities
│   ├── models/               # Inception-V3 & ResNet-18 setup
│   └── utils/                # Helpers, seed control, plotting
│
├── notebooks/
│   └── emotion_classification.ipynb  # Full experiments, analysis, plots
│
├── outputs/
│   ├── best_model.pth        # Saved checkpoint
│   ├── learning_curves.png   # Training/validation graphs
│   └── confusion_matrix.png
│
└── README.md

---

## Dataset
- **Total images:** 1,148  
- **Classes:** 6 balanced emotions  
- **Split:** 70% train, 15% validation, 15% test  
- **Sampling:** Stratified to preserve class proportions

##  Training the Models

### **1. Install dependencies**
```bash
pip install torch torchvision scikit-learn matplotlib numpy pillow
````

### **2. Train baseline (ResNet-18)**

```bash
python src/train.py --model resnet18 --augment false
```

### **3. Train with augmentation**

```bash
python src/train.py --model resnet18 --augment true
```

### **4. Train Inception-V3**

```bash
python src/train_inception.py
```

### **5. Evaluate on test set**

```bash
python src/evaluate.py
```

---

## Results Summary

### **Inception-V3 (Transfer Learning)**

* **Accuracy:** ~51%
* **Macro-F1:** ~0.51
* **Macro ROC-AUC:** ~0.84

Strong discrimination ability but limited by dataset size.

---

### **Baseline ResNet-18 (No Augmentation)**

* **Accuracy:** 57.8%
* **Macro-F1:** 0.586
* **Macro ROC-AUC:** 0.86

---

### **ResNet-18 With Augmentation**

* **Accuracy:** 58.4%
* **Macro-F1:** 0.577
* Marked improvement for challenging class **“anger”**
* More stable training curves, less overfitting

---

##  Overfitting & Sanity Checks

* Learning curves show moderate overfitting in baseline models.
* **Randomized-label sanity test** confirms model does *not* memorize data.
* Augmentation slows early learning but improves robustness on difficult classes.

---

##  Future Improvements

* Fine-tuning deeper layers of Inception-V3
* Using Vision Transformers (ViT / Swin)
* Applying focal loss for hard classes (pain, sad)
* Face alignment preprocessing
* Larger curated dataset


