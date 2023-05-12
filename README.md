# EMV-3D-CNN

This repository contains the code of **An Ensemble Multi-View 3D Convolution Neutral Network Model for Lung Adenocarcinoma Risk Stratification on Thin Slice Computed Tomography: A Multi-Center Study**. 

## Introduction

Lung cancer has always been among the most frequently diagnosed cancers threatening people’s health worldwide. However, seldom studies predict the invasive grades of lung adenocarcinoma. This is an important task since it will be helpful to design a more reasonable surgical mode (lobectomy or sublobar resection) before operation. We propose an ensemble multi-view 3D convolution neutral network (EMV-3D-CNN) model to comprehensively study the risk grades of lung adenocarcinoma. 

## Model
Figure 1 shows the flowchart of the proposed EMV-3D-CNN model. Our approach consists of three key tasks. They are, respectively, diagnosing benign and malignant lung tumors (Task 1), classifying between pre-invasive and invasive lung tumors (Task 2), and identifying the risk stratification (i.e, Grade 1, Grade 2, Grade 3) of invasive lung tumors (Task 3).
![model_flowchart.png](https://github.com/zhoujing89/EMV-3D-CNN/blob/main/images/model_flowchart.png?raw=true)

Finally, for user-friendly access, the proposed model is also implemented as a web-based system (https://seeyourlung.com.cn). By uploading the full original CT images of DICOM format, our algorithm can give the risk grades of pulmonary nodules by specifying the center location of the target lung nodule.


