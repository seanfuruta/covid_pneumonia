Identifying Types of Pneumonia With Chest X-Rays From COVID-19 Exams
W281 - Computer Vision
Final Project Proposal
Rakesh Walisheter, Sean Furuta, David Miller
Introduction
In the wake of the COVID-19 pandemic, we want to use chest x-ray imagery to improve predictions of infection to better manage patient care. A large collection of x-ray image data was collected for patients with potential infection, allowing for advanced computer vision techniques to distinguish between COVID-19 and traditional pneumonia.
Data Description
Chest X-Ray Dataset - https://darwin.v7labs.com/v7-labs/covid-19-chest-x-ray-dataset

The dataset contains 6,500 unique x-ray images, each containing the following annotations:
2 masks (one for each lung)
Tag for type of pneumonia (viral, bacterial, fungal, healthy/none)
Tag for COVID-19
If COVID-19 == Yes
Tag for age, sex, temperature, location, intubation status, ICU admission, and patient outcome

We expect some level of data preparation necessary to reach a suitable modeling dataset based on the following factors:
Image resolution varies from 5600x4700 to 156x156
Lateral x-rays do not contain lung segmentations
63 axial CT scan slices without masks
Portable x-ray images are low quality
Some images are obfuscated by medical instruments

Example images from the dataset shown below:



Feature Extraction
We will employ several feature extraction methods to enrich the image data for classification.

Method
Description
Feature pre-processing


Histogram equalization
Extend the pixel’s intensity range from the original range to 0 to 255. The enhanced image has a wider range of intensity and slightly higher contrast.
Adaptive masking
Create an adaptive mask that after bitwise operation removes the diaphragm from the source image.
Gaussian blur
Filter reduces some noise and unwanted details
Non-learned features


Edge detection


Luminance values (histogram)


Blob detection


Learned features


ORB
Oriented FAST and rotated BRIEF feature detector and binary descriptor extractor.
SIFT
SIFT feature detection and descriptor extraction.
DAISY
Extract DAISY feature descriptors densely for the given image.

Classification Problem
The goal of the project is to identify the type of pneumonia using x-ray images of the lungs. This is a multi-class classification problem that will distinguish between viral, bacterial, fungal, and no (i.e.: healthy) pneumonia.

We will build the following 3 multi-class classification models and evaluate each using a balanced accuracy measure:

Logistic Regression (baseline)
Gradient Boosting Classifier
Convolutional Neural Network

This approach will allow us to interpret the incremental gain in model performance by moving from a basic to advanced traditional model (i.e.: logistic regression to GBT) and from traditional ML to deep learning (i.e. GBT to CNN).

References
Joseph Paul Cohen, Paul Morrison, and Lan Dao. 2020. COVID-19 Image Data Collection. https://arxiv.org/pdf/2003.11597.pdf
