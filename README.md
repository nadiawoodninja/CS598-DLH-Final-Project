# CS598 DLH Final Project Spring 2022
## Nadia Wood (nadiaw2) and Diana Gonzalez Santillan (dianag4)


#### Python version: Python3 
#### IDE: Visual Studio Code 
In order to run the code provided by the authors of the paper, we had to install the following dependencies. 
We used Visual Studio Code to import the project into and then proceeded to install the required libraries
<br>
  `sudo pip3 install torch torchvision`
<br>
  `sudo pip3 install Cython`
<br>
  `sudo pip3 install torchsparseattn`

Once the dependencies are installed run the code by executing this
<br>
`python3 main.py`

## Open Datasets
We used UC Irvine Machine Learning Repository to acquire our datasets needed for the project. This repository can be accessed here: https://archive.ics.uci.edu/ml/index.php

### Here are the datasets that we used from the repository

Heart Disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
Diabetes Disease : https://archive.ics.uci.edu/ml/datasets/Diabetes
Kidney Disease : https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

## MIMIC III Demo Dataset
We used these instructions to access the MIMIC III demo data: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/
`SELECT  * FROM `physionet-data.mimiciii_demo.admissions` `
