# EMV-3D-CNN

This repository contains the code for **An Ensemble Multi-View 3D Convolution Neural Network Model for Lung Adenocarcinoma Risk Stratification on Thin Slice Computed Tomography: A Multi-Center Study**.

## Contents

- [Introduction](#introduction)
- [Model](#model)
- [Deployment](#Deployment)
- [Requirements](#requirements)
- [Installation](#Installation)

A detailed tutorial video can be found below to guide users through our web-based system at [seeyourlung.com.cn](https://seeyourlung.com.cn).



## Introduction

Lung cancer is among the most frequently diagnosed cancers worldwide. However, few studies predict the invasive grades of lung adenocarcinoma, an important task that can assist in planning a suitable surgical approach (lobectomy or sublobar resection) prior to operation. 

We propose an ensemble multi-view 3D convolution neural network (EMV-3D-CNN) model to comprehensively study the risk grades of lung adenocarcinoma. Our codes consist of three main parts: preprocessing, training, and evaluation.

## Model

Figure 1 shows the flowchart of the proposed EMV-3D-CNN model. It involves three key tasks: diagnosing benign and malignant lung tumors (Task 1), classifying between pre-invasive and invasive lung tumors (Task 2), and identifying the risk stratification (i.e., Grade 1, Grade 2, Grade 3) of invasive lung tumors (Task 3).






![model_flowchart.png](https://github.com/zhoujing89/EMV-3D-CNN/blob/main/images/model_flowchart.png?raw=true)

We provide the trained models for various 3D medical image analysis tasks, which can be downloaded from [BaiduYun](https://pan.baidu.com/s/1Y7WuAdBSMcbuBlf943l04Q?pwd=3acd)(verification code: 3acd).

## Deployment

For easier access, we have also implemented the proposed model as a web-based system at [seeyourlung.com.cn](https://seeyourlung.com.cn). By uploading the full original CT images in DICOM format, our algorithm can assign risk grades to pulmonary nodules given the center location of the target lung nodule. 


![platform.png](https://github.com/zhoujing89/EMV-3D-CNN/blob/main/images/platform.png?raw=true)

## Requirements

The code is written in Python and requires the following packages: 

* Python 3.9.12 
* TensorFlow 2.8.0 
* Keras 2.8.9 
* Matplotlib 3.5.2 
* Numpy 1.22.4 
* Pandas 1.4.2 
* Sklearn 1.1.1 
* Scipy 1.8.1
## Installation
* Install Python 3.9.12
* pip install -r requirements.txt
