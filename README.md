# Deep-Learning-Digit-Recognizer
A CNN-based model for recognizing handwritten digits using the MNIST dataset.


This project implements a **Convolutional Neural Network (CNN)** model to recognize handwritten digits using the **MNIST dataset**. The goal of the project is to classify images of handwritten digits (0-9) with high accuracy. The MNIST dataset is a standard dataset in the field of machine learning and deep learning, consisting of 60,000 training images and 10,000 test images.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The **Deep Learning Digit Recognizer** project uses a CNN model to classify handwritten digits from the MNIST dataset. The model is trained on 60,000 images and tested on 10,000 images. The project demonstrates the use of deep learning techniques for image classification tasks.

Key features:
- Implemented using **Keras** with TensorFlow backend.
- Achieved an accuracy of **96.17%** on the test set.
- Final submission to Kaggle competition yielded an accuracy score of **98.93%**.

---

## Technologies Used

This project uses the following technologies and libraries:
- **Python 3.x**
- **Keras** (for building and training the CNN model)
- **TensorFlow** (as the backend for Keras)
- **NumPy** (for data manipulation)
- **Matplotlib** (for data visualization)

---

## Dataset

The project uses the **MNIST dataset**, which consists of grayscale images of handwritten digits (0 to 9). Each image is a 28x28 pixel square.

### Dataset Details:
- Training set: 60,000 images
- Test set: 10,000 images
- Image size: 28x28 pixels
- Number of classes: 10 (digits from 0 to 9)

The dataset can be downloaded directly using Keras:

```python
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
