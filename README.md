# CIFAR-10 Image Classification with PyTorch CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project demonstrates image classification using a custom CNN architecture built with PyTorch. The model is trained on the CIFAR-10 dataset to classify images into 10 different categories. The implementation includes data preprocessing, model training, evaluation, and inference on custom images.

## 📊 Dataset

**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:
- ✈️ Airplane
- 🚗 Car  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚛 Truck

- **Training set**: 50,000 images
- **Test set**: 10,000 images
- **Image dimensions**: 32×32×3 (RGB)

## 🏗️ Model Architecture

The CNN architecture consists of:

### Convolutional Layers
- **Conv1**: 3→12 channels, 5×5 kernel, ReLU activation
- **MaxPool1**: 2×2 pooling
- **Conv2**: 12→24 channels, 5×5 kernel, ReLU activation  
- **MaxPool2**: 2×2 pooling

### Fully Connected Layers
- **FC1**: 600→120 neurons, ReLU activation
- **FC2**: 120→84 neurons, ReLU activation
- **FC3**: 84→10 neurons (output layer)

**Total Parameters**: ~62,000

## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for faster training)

### Dependencies
```bash
pip install torch torchvision numpy pillow jupyter
```

### Clone Repository
```bash
git clone https://github.com/NathanCordeiro/cifar10-image-classification.git
cd cifar10-image-classification
```

## 💻 Usage

### 1. Training the Model
Run the Jupyter notebook `main.ipynb` or execute the training script:

```python
# The model trains for 30 epochs with the following configuration:
# - Optimizer: SGD with learning rate 0.001, momentum 0.9
# - Loss function: CrossEntropyLoss
# - Batch size: 32
```

### 2. Model Inference
To classify your own images:

```python
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load and preprocess image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load trained model
net = NeuralNet()
net.load_state_dict(torch.load('cifar10_neural_net.pth'))
net.eval()

# Classify image
image = Image.open('your_image.jpg')
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = net(image_tensor)
    _, predicted = torch.max(outputs.data, 1)
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f'Predicted class: {class_names[predicted.item()]}')
```

### 3. Model Evaluation
The trained model achieves **68.90% accuracy** on the CIFAR-10 test set.

## 📈 Results

### Training Progress
- **Initial Loss (Epoch 0)**: 2.1473
- **Final Loss (Epoch 29)**: 0.4770
- **Test Accuracy**: 68.90%

### Performance Metrics
| Metric | Value |
|--------|-------|
| Test Accuracy | 68.90% |
| Training Epochs | 30 |
| Batch Size | 32 |
| Learning Rate | 0.001 |

## 📁 Project Structure

```
cifar10-image-classification/
│
├── main.ipynb                 # Main Jupyter notebook
├── cifar10_neural_net.pth    # Trained model weights
├── data/                     # CIFAR-10 dataset (auto-downloaded)
├── IMG1.jpg                  # Sample test images
├── IMG2.jpeg
├── IMG3.jpg
└── README.md                 # Project documentation
```

## 🔧 Model Details

### Data Preprocessing
- **Normalization**: RGB values normalized to [-1, 1] range using mean=0.5, std=0.5
- **Tensor Conversion**: PIL images converted to PyTorch tensors
- **Resizing**: Custom images resized to 32×32 pixels

### Training Configuration
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 30

### Model Features
- **Activation Function**: ReLU for non-linearity
- **Pooling**: Max pooling for dimensionality reduction
- **Regularization**: Implicit through architecture design
- **Output**: 10-class probability distribution

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009
- **PyTorch Team**: For the excellent deep learning framework
- **torchvision**: For dataset utilities and transforms

---

**Note**: This model serves as a learning example for CNN implementation. For production use, consider more advanced architectures like ResNet, DenseNet, or EfficientNet for better performance.