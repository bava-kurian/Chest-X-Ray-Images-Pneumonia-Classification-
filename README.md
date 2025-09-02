🩻 Chest X-Ray Pneumonia Classifier
📌 Project Overview

This project uses deep learning to classify chest X-ray images as Normal or Pneumonia.
We leverage Transfer Learning (ResNet50) to achieve high accuracy on medical images while keeping training efficient.

📂 Dataset

Source: Chest X-Ray Images (Pneumonia) – Kaggle

Split:

Train: 5216 images

Validation: 16 images

Test: 624 images

Classes: Normal, Pneumonia

Images resized to 224×224, normalized to [0,1]

⚙️ Preprocessing

Rescale pixel values

Resize all images to 224×224

Data Augmentation: rotation, zoom, horizontal flip

🧠 Model Architecture

Base Model: ResNet50 (pretrained on ImageNet, frozen)

Custom Layers:

Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Loss Function: Binary Crossentropy

Optimizer: Adam (lr=0.0001)

Metrics: Accuracy

🚀 Training Setup

Batch Size: 32

Epochs: 10 (can be tuned)

Hardware: GPU recommended (Google Colab/Cloud VM)

📊 Results

Test Accuracy: ~85–92%

Strong ability to distinguish Pneumonia vs Normal

Best practice: evaluate with Precision, Recall, F1-score, ROC-AUC

📈 Visualizations

Training & validation accuracy/loss curves

(Optional) Grad-CAM to visualize important regions in X-rays

💾 Deployment

Save trained model:

model.save("pneumonia_classifier.h5")


Deploy as:

Flask/FastAPI backend

Streamlit dashboard

TensorFlow Lite for mobile apps

🔮 Future Improvements

Fine-tuning upper ResNet layers

Handling class imbalance with class_weight or oversampling

Testing stronger models (DenseNet, EfficientNet)

Adding explainability with Grad-CAM