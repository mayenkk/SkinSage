SkinSage

SkinSage is a deep learning project aimed at predicting skin diseases from images. This repository contains code for training and deploying a model that can classify skin diseases based on input images.

Overview

In this project, we utilize a pre-trained model ResNet50 to build a skin disease classification model. The model has been trained on a dataset comprising over 24,000 images of various skin diseases, covering 24 different skin conditions. The trained model achieves an accuracy score of 91.7% on the test set.

Model Architecture

The model architecture consists of the following components:

ResNet50: A pre-trained convolutional neural network known for its effectiveness in image classification tasks.
Global Average Pooling Layer: Added on top of the ResNet50 base to reduce the spatial dimensions of the feature maps.
Dense Layers: Two dense layers are added for classification purposes.
Early Stopping: Implemented to prevent overfitting and to halt training when the model performance on the validation set stops improving.
Usage

To use this model, follow these steps:

Installation: Clone this repository to your local machine.
bash
Copy code
git clone https://github.com/your-username/SkinSage.git
Dependencies: Make sure you have the required dependencies installed. You can install them using:
bash
Copy code
pip install -r requirements.txt
Dataset: Ensure that you have access to the dataset of skin disease images.
Training: Run the training script to train the model.
bash
Copy code
python train.py --dataset <path_to_dataset> --epochs <num_epochs> --batch_size <batch_size>
Evaluation: Evaluate the trained model on the test set.
bash
Copy code
python evaluate.py --model <path_to_saved_model> --test_data <path_to_test_data>
Prediction: Use the trained model to make predictions on new images.
bash
Copy code
python predict.py --model <path_to_saved_model> --image <path_to_image>
Acknowledgements

This project was made possible by the contributions of many individuals. We would like to thank all the contributors and the open-source community for their valuable input and support.

License

This project is licensed under the MIT License - see the LICENSE file for details.
