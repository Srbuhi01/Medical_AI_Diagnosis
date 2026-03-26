# 🩺 Medical AI Diagnosis  
## Automatic Disease Detection from Chest X-rays using CNNs and Transformers  

---

## 📌 Project Overview  
This project focuses on automatic detection of thoracic diseases from chest X-ray images using deep learning models.  
The goal is to build a **multi-label classification system** and compare the performance of **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViT)**.

The project is part of an ongoing research-oriented study aimed at understanding how modern deep learning architectures perform on real-world medical datasets.

---

## 📊 Dataset  
- **Dataset:** ChestX-ray14 (NIH)  
- **Task:** Multi-label classification  
- **Number of classes:** 14  

**Diseases include:**  
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax  

---

## 🤖 Models  

### 🔵 CNN Model  
- **Architecture:** ResNet50 (pretrained)  
- Modified final layer for 14-class classification  

### 🔴 Transformer Model  
- **Architecture:** Swin Transformer (Tiny)  
- Fine-tuned for medical image classification  

---

## ⚙️ Training Details  
- **Framework:** PyTorch  
- **Loss Function:** BCEWithLogitsLoss (with class imbalance handling via `pos_weight`)  
- **Optimizers:**  
  - CNN → Adam  
  - Transformer → AdamW  

### Techniques Used  
- Mixed Precision Training (FP16)  
- Early Stopping  
- Gradient Clipping (Transformer)  

### Data Augmentation  
- Resize (224×224)  
- Random Horizontal Flip  
- Random Rotation  

---

## 📈 Current Results (Work in Progress)  

### 🧠 ResNet50 (CNN)  
- **Average AUC:** 0.7739  

### 🔥 Swin Transformer  
- **Average AUC:** 0.7847  

👉 Transformer currently shows slightly better performance across most classes.

⚠️ *Note: Experiments are ongoing and results may improve with further tuning and additional models.*

---

## 📊 Example Results (AUC per Class)

| Disease              | CNN (ResNet50) | Transformer (Swin) |
|---------------------|---------------|--------------------|
| Cardiomegaly        | 0.8484        | 0.8591             |
| Emphysema           | 0.8600        | 0.8964             |
| Pneumothorax        | 0.8222        | 0.8467             |
| Infiltration        | 0.6845        | 0.6935             |
| Pneumonia           | 0.6761        | 0.6996             |


---

## 🔍 Key Features  
- Multi-label classification on real-world medical dataset  
- Handling severe class imbalance using weighted loss  
- Comparison between CNN and Transformer architectures  
- Grad-CAM visualization for model interpretability  
- Training logs and performance tracking  

---

## 🔬 Ongoing Work  
- Testing additional architectures (EfficientNet, ViT variants)  
- Hyperparameter tuning  
- Improving performance on minority classes  
- Exploring ensemble methods  

