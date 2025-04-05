# Hand Sign Detection Using Deep Learning ##
A real-time computer vision system that recognizes hand signs using Convolutional Neural Networks.

## 📝 Overview ##
This project implements a deep learning model to detect and classify hand signs in real-time using a webcam. It can recognize different hand gestures (numbers 0-5) and could be extended to support sign language recognition or gesture-based interfaces.

## ✨ Features

1) Real-time hand sign detection and classification
2) Custom CNN architecture optimized for gesture recognition
3) Data augmentation to improve model generalization
4) Interactive webcam interface for live testing
5) Complete data collection pipeline for creating custom datasets
6) Performance visualization and model evaluation tools

## 🛠️ Technologies

1) TensorFlow & Keras
2) OpenCV
3) NumPy & Pandas
4) Matplotlib
5) Google Colab integration

## 🏗️ Project Structure
hand-sign-detection/

│

├── data/

│   ├── train/                 # Training images sorted by category

│   └── test/                  # Test images for evaluation

│

├── models/                    # Saved model files

│

├── src/

│   ├── data_collection.py     # Scripts for data collection

│   ├── data_processing.py     # Data preprocessing and augmentation

│   ├── model.py               # CNN model architecture

│   └── prediction.py          # Real-time prediction module

│

├── notebooks/                 # Jupyter notebooks for experimentation

│   └── hand_sign_detection.ipynb

│

├── README.md                  # Project documentation

└── requirements.txt           # Project dependencies


## 🚀 Getting Started

## Prerequisites

1) Python 3.7+
2) TensorFlow 2.x
3) OpenCV 4.x
4) Webcam (for real-time testing)

## Installation

1) Clone this repository:
   
   git clone https://github.com/Saloni1519/Handsign-detection-AI
   
   cd hand-sign-detection
   
2) Install dependencies:
   
   pip install -r requirements.txt

3) Download pre-trained model or train your own:
   
   python src/model.py --train

## 📊 Model Architecture
The model uses a convolutional neural network (CNN) with the following architecture:

1) 3 convolutional layers with increasing filter sizes (32, 64, 128)
2) Max pooling after each convolutional layer
3) Dropout regularization to prevent overfitting
4) Dense layers for classification with softmax activation

## 🔍 Implementation Details
The project follows a systematic approach:

1) Data Collection: Custom function to collect hand sign images via webcam
2) Preprocessing: Image normalization, resizing, and color conversion
3) Data Augmentation: Random rotations, shifts, zooms to improve generalization
4) Model Training: Training with batched data and validation
5) Evaluation: Model metrics and performance visualization
6) Real-time Inference: Live webcam integration for interactive testing

## 📈 Results
The model achieves XX% accuracy on the test set, with real-time 
inference capable of running at YY FPS on standard hardware.

## 🔮 Future Improvements
1) Expand to full ASL alphabet recognition
2) Implement hand tracking for gesture sequences
3) Explore transfer learning with pre-trained models
4) Optimize for mobile deployment








