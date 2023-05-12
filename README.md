# EMV-3D-CNN
This repository contains the code of **An Ensemble Multi-View 3D Convolution Neutral Network Model for Lung Adenocarcinoma Risk Stratification on Thin Slice Computed Tomography: A Multi-Center Study**. 

## Introduction

Lung cancer has always been among the most frequently diagnosed cancers threatening people’s health worldwide. However, seldom studies predict the invasive grades of lung adenocarcinoma. This is an important task since it will be helpful to design a more reasonable surgical mode (lobectomy or sublobar resection) before operation. We propose an ensemble multi-view 3D convolution neutral network (EMV-3D-CNN) model to comprehensively study the risk grades of lung adenocarcinoma. Our model achieves a state-of-art performance (91.3% AUC for diagnosis between benign and malignant, 92.5% AUC for diagnosis between pre-invasive and invasive, and 77.6% accuracy for diagnosis among risks of Grades 1,2,3) on 1,075 lung nodules (covering 627 CT trials) collected from three medical centers. 

## Model Development


Finally, for user-friendly access, the proposed model is also implemented as a web-based system (https://seeyourlung.com.cn). By uploading the full original CT images of DICOM format, our algorithm can give the risk grades of pulmonary nodules by specifying the center location of the target lung nodule.


