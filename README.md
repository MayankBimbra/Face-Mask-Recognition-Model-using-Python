# Neural Networks & Deep Learning Project--
# Real Time Face-Mask-Recognition-Model using a Web Interface in Python

# Description
Face masks are crucial in minimizing the propagation of Covid-19, and are highly recommended or even obligatory in many situations. In this project, we develop a pipeline to detect unmasked faces in images. This can, for example, be used to alert people that do not wear a mask when entering a building.
This project consists of three steps:
1. Detect all human faces in an real time using web camera.
2. Make a with_mask/without_mask prediction for each of them
3. Return an annotated model with the predictions

# Problem
#### For any real time image taken from our web camera, our goal is to predict that the person is wearing a Face Mask or not.
 i.e. CLASSIFY WHETHER THE PERSON IS WEARING A MASK OR NOT. 
# Project Formulation

The hands on building this project of Face Mask Recognition Model is divided into following tasks/steps:-

#### A.	Task 1: Introduction 
•	Introduction to the dataset

•	Import essential modules and helper dependencies.

#### B.	Task 2: Exploring the Dataset

#### C.	Task 3: Generating Training and Validation Batches

#### D.	Task 4: Creating a Convolutional Neural Network (CNN) Model
•	Design a convolutional neural network with 4 convolution layers and 2 fully connected layers to predict.

•	Used Adam as the optimizer, categorical crossentropy as the loss function, and accuracy as the evaluation metric.

#### E.	Task 5: Training and Evaluating Model
•	Training the CNN by invoking the model.fit() method.

•	Used ModelCheckpoint() to save the weights associated with the higher validation accuracy.

#### F.	Task 6: Saving and Serializing Model as JSON String
•	Used to_json(), which uses a JSON string, to store the model architecture.

#### G.	Task 7: Creating a Frame to Serve Predictions

#### H.	Task 8: Creating a Class to Output Model Predictions

#### I.	Task 10: Used Model to Recognize Face Mask on not at the Real Time using laptops webcamera
•	We than run the main.py script to create the Frame and serve the model's predictions to a web interface.

•	Applied the model for real time recognition of Face Mask of users using webcam of the Laptop.


# Dataset Description
Downloadable Link - https://drive.google.com/drive/folders/1Dm2sV8UrMd6OKzjVkW859WznhfSXFZF8
The dataset used in this project work has been taken from the Kaggle.com i.e. a Faces dataset(with/without mask Dataset).

The dataset consists of 3,833 images belonging to two classes:
with_mask: 1915 images
without_mask: 1918 images

# Prerequisites
You need to have installed following softwares and libraries in your machine before running this project.

1. tensorflow>=1.15.2
2. keras==2.3.1
3. imutils==0.5.3
4. numpy==1.18.2
5. opencv-python==4.2.0.*
6. matplotlib==3.2.1
7. scipy==1.4.1

# Steps followed for building the project

As per various surveys it is found that for implementing this project four basic steps are required to be performed.

i.) Preprocessing- Preprocessing is a common name for operations with images at the lowest level of abstraction both input and output are intensity images. 

ii.) Face registration- Face Registration is a computer technology being used in a variety of applications that identifies human faces in digital images. 

iii.) Facial feature extraction- Facial Features extraction is an important step in face recognition and is defined as the process of locating specific regions, points, landmarks, or curves/contours in a given 2-D image or a 3D range image.  

iv.) Applying Face Mask Detector Model.

# Conclusion
To mitigate the spread of COVID-19 pandemic, measures must be taken. I have modeled a face mask detector using learning methods in neural networks. To train, validate and test the model, I used the dataset that consisted of 1915 masked faces images and 1918 unmasked faces images. These images were taken from various resources like Kaggle and RMFD datasets. The model was inferred on images and live video streams.
To select a base model, we evaluated the metrics like accuracy, precision and recall and selected MobileNetV2 architecture with the best performance having 100% precision and 99% recall. It is also computationally efficient using MobileNetV2 which makes it easier to install the model to embedded systems.
This face mask detector can be deployed in many areas like shopping malls, airports and other heavy traffic places to monitor the public and to avoid the spread of the disease by checking who is following basic rules and who is not.

# Author
Mayank Bimbra
