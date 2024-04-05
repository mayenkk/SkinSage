# **SkinSage**

SkinSage is a deep learning project aimed at predicting skin diseases from images. This repository contains code for training and deploying a model that can classify skin diseases based on input images.

## **Overview**

In this project, we utilize a pre-trained model ResNet50 to build a skin disease classification model. The model has been trained on a dataset comprising over 24,000 images of various skin diseases, covering 24 different skin conditions. The trained model achieves an accuracy score of 91.7% on the test set.

## **Model Architecture**

The model architecture consists of the following components:

1. **ResNet50**: A pre-trained convolutional neural network known for its effectiveness in image classification tasks.
2. **Global Average Pooling Layer**: Added on top of the ResNet50 base to reduce the spatial dimensions of the feature maps.
3. **Dense Layers**: Two dense layers are added for classification purposes.
4. **Early Stopping**: Implemented to prevent overfitting and to halt training when the model performance on the validation set stops improving.


