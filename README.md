# Image Scene Classification using Convolutional Neural Networks

## Overview

This project aims to classify images of natural scenes into six categories: buildings, forest, glacier, mountain, sea, and street. The dataset consists of around 25,000 images of size 150x150, with approximately 14,000 images for training, 3,000 for testing, and 7,000 for prediction.

## Dataset

The dataset used for this project is sourced from Kaggle and can be found [here](https://www.kaggle.com/datasets/puneet6060/intel-image-classification/data). It contains images of natural scenes from various locations around the world, distributed across the following categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

## Obtaining Kaggle API Token

To download the dataset directly from Kaggle, you need to have a Kaggle API token (`kaggle.json`). Here's how you can obtain and upload it:

1. Go to the Kaggle website and sign in (or sign up if you don't have an account).
2. Navigate to your account settings by clicking on your profile picture in the top-right corner and selecting "Account".
3. Scroll down to the API section and click on "Create New API Token". This will download a file named `kaggle.json`.
4. Click the "Choose File" button below to upload your `kaggle.json` file.

## Model Architecture

We employed a Convolutional Neural Network (CNN) for image classification. The model architecture consists of multiple convolutional layers followed by max-pooling layers to extract relevant features from the images. After flattening the output, dense layers are used for classification, with a softmax activation function to output probabilities for each class.

## Implementation

The dataset was loaded using the Image Data Generator from TensorFlow, which allows for efficient handling of image data during training. Data augmentation techniques such as rescaling, zooming, and horizontal flipping were applied to the training images to improve model generalization.

The CNN model was trained using the Adam optimizer and categorical cross-entropy loss function. Early stopping was employed to prevent overfitting, and model checkpoints were used to save the best-performing model during training.

## Evaluation

The model's performance was evaluated using classification metrics such as accuracy, precision, recall, and F1-score. Additionally, a confusion matrix was generated to visualize the model's performance across different classes. Wrong predictions were identified and displayed along with the corresponding true and predicted labels to analyze the model's errors.

## Requirements

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
