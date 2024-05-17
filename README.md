# Identifying Types of Pneumonia with Chest Xray Images
#### Sean Furuta   ,   David Miller   ,   Rakesh Walisheter

## Introduction

Chest X-Ray (CXR) images have been an essential tool to diagnose pneumonia in patients [[Ref-1](#References)]. 

In the wake of the COVID-19 pandemic, A large curated dataset of CXR images for patients with potential pneumonia-infections, and their corresponding lung-secgementation masks has been created by the authors of the referenced paper. This dataset now allows for advanced computer vision techniques to distinguish between the novel COVID-19 and other types of pneumonia.

This study intends to use this CXR Image dataset to build models which can classify a CXR as either Normal, or infected with COVID19 or infected with a non-COVID Pneumonia.

## Executive Summary

To prepare CXR images for modelling a preprocessing pipe with these steps was developed:

1. Histogram equalization
2. Contrast limited adaptive histogram equalization (CLAHE)
3. Local normalization
4. Denoising

To reduce dimensionality while retaining as much relevant information as possible, a feature engineering pipeline was designed with the following steps:

1. Scale Invarient Feature Transform(SIFT) [Ref-3] keypoints detection.
2. Cluster keypoints from SIFT into N-Clusters (N=10 for this study) based on position and size.
3. Use the cluster labels to then calculate summary statistics (mean, variance, standard-deviation) for the Position (x,y) and Size.
4. Texture features using Gray-Level Co-occurance Matrix (GLCM) features.
5. Texture features using Local Binary Patterns.
6. Flatten and normalize each feature-vector to floating point vectors between 0-1.
7. Concatenate into the final feature vector.

We developed 4 distinct models to evaluate the image classification task:

| Model | Accuracy |
| --- | --- |
| Logistic Regression | .543 |
| Gradient Boosted Classifier | .671 |
| RESNET50 | .886 |
| CovXNET | .905 |

## Repository Structure

- [Final Report PDF](./FinalReport.pdf) 
- [Final Report Notebook](./FinalReport.ipynb) 
- Detailed EDA and code notebooks can be found under `./code`
