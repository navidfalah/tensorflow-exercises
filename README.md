# 🧠 Deep Learning Image Classification Projects

## 🚀 Projects Overview
This repository contains two deep learning projects focused on image classification using TensorFlow and Keras:

1. **Horse vs. Human Classification** 🐎👨‍💻 - A binary classification model that differentiates between images of horses and humans.
2. **Rock-Paper-Scissors Gesture Recognition** ✊✋✌️ - A multi-class classification model that recognizes hand gestures for rock, paper, and scissors.

---

## 🐎 Horse vs. Human Classification
### 📌 Overview
This project trains a Convolutional Neural Network (CNN) to classify images as either a horse or a human.

### 📊 Dataset
The dataset is sourced from Kaggle and contains images in separate directories for training and validation.

### ⚙️ Model Architecture
- Convolutional layers with increasing filter sizes
- Max-pooling layers to reduce spatial dimensions
- Fully connected dense layers with ReLU activation
- Sigmoid activation for binary classification

### 🚀 Training & Evaluation
- Uses **ImageDataGenerator** for real-time data augmentation
- **Binary cross-entropy loss** with **RMSprop optimizer**
- Early stopping when accuracy reaches 99%

### ▶️ Prediction
Users can upload an image, and the trained model will predict if it is a horse or a human.

### 📂 Model Deployment
The trained model is saved as `hh.keras` and uploaded to Hugging Face.

---

## ✊ Rock-Paper-Scissors Gesture Recognition
### 📌 Overview
This project trains a CNN to classify hand gestures into three categories: rock, paper, or scissors.

### 📊 Dataset
The dataset is sourced from Kaggle and consists of images labeled for the three gestures.

### ⚙️ Model Architecture
- Multiple convolutional layers with ReLU activation
- Max-pooling layers to downsample features
- Dense layers for final classification
- **Softmax activation** for multi-class classification

### 🚀 Training & Evaluation
- **Categorical cross-entropy loss** with **RMSprop optimizer**
- Data augmentation applied to training images

### ▶️ Prediction
Users can upload an image, and the model will predict the corresponding hand gesture.

### 📂 Model Deployment
The trained model is saved as `rps_model.h5` and uploaded to Hugging Face.

---

## 👤 Author
Developed by Navid Falah.
