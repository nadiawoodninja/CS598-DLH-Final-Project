# CS598 DLH Final Project Spring 2022
## Nadia Wood (nadiaw2) and Diana Gonzalez Santillan (dianag4)

Work based on paper by ... ... 
TODO -- include paper citation

Code adaptation from original code sent to us by authors, downloadable here:
TODO -- add link to google drive link the author sent to us with original code.

### Running the code

#### Python version: Python3 
#### IDE: Visual Studio Code 
In order to run the code provided by the authors of the paper, we had to install the following dependencies. 
We used Visual Studio Code to import the project, and then intalled the following required libraries:
<br>
  `sudo pip3 install torch torchvision`
<br>
  `sudo pip3 install Cython`
<br>
  `sudo pip3 install torchsparseattn`
<br>
  `sudo pip3 install pandas`

Once the dependencies are installed you can run the preprocessing, training, and evaluation code by executing the follwing command:
<br>
`python3 main.py --task=[TASK] --emb_dim=256 --d_k=256 --d_v=256 --d_inner=256 --use_cuda=False --optimizer=Adam`
<br>
where `[TASK]` is one of 'diagnoses', 'heart', 'diabetes', or 'kidney'. Feel free to change any of the other hyperparameters to see the changes in the results of the model. Here is the full list of options available for you to run the model, along with their default values:

--task, choices=('diagnoses', 'heart', 'diabetes', 'kidney'), default='heart', task to run
--data_root, type=str, default='../datasets/', dataset root directory
--fold, choices=(1, 2, 3, 4, 5), default=1, number of fold
--use_cuda, action='store_true', default=False, if use GPU
--gpu_devices, type=str, default='0', device IDs for GPU
--epochs, type=int, default=25, number of epochs
--batch_size, type=int, default=32
--drop_rate, type=float, default=0.5
--optimizer, choices=('Adam', 'SGD', 'Adadelta'), required=False
--lr, type=float, default=5e-4, learning rate
--weight_decay, type=float, default=1e-4
--n_head, type=int, default=2, number of head of self-attention for the visit attention
--n_depth, type=int, default=2, number of layers of self-attention for the visit attention
--emb_dim, type=int, default=128, size of medical variable (or code) embedding.
--d_k, type=int, default=128, size of vector before self attention
--d_v, type=int, default=128, size of vector before self attention
--d_inner, type=int, default=128
--dvp, action='store_true', default=False, Weather use position embedding
--dp, action='store_true', default=False, Weather use position embedding
--ds, action='store_true', default=False, whether delete the sparse_max
--cap_uncertainty, action='store_true', Weather capture uncertainty, default=True
--monto_carlo_for_aleatoric, type=int, default=100, size of Monto Carlo Sample
--monto_carlo_for_epistemic, type=int, default=200, size of Monto Carlo Sample
--analysis_dir, type=str, default='../../output_for_analysis_final/', nalysis output dir
--write_performance, action='store_true', default=False, Weather write performance result
--performance_dir, type=str, default='../../metric_results/', performance dir
--save_model_dir, type=str, default='../../saved_model', set dir to save the model which has the best performance
--resume, type=str, default=None, Choose the model dict to load for test or fine-tune.
--data_scale, default=1, type=float

<br>
If you plan to use GPU computation, install CUDA: https://developer.nvidia.com/cuda-downloads and include the --use_cuda=True flag.

## Open Datasets

TODO -- data download instructions

We used UC Irvine Machine Learning Repository to acquire our datasets needed for the project. This repository can be accessed here: https://archive.ics.uci.edu/ml/index.php

### Here are the datasets that we used from the repository

Heart Disease: https://archive.ics.uci.edu/ml/datasets/Heart+Disease <br>
Diabetes Disease : https://archive.ics.uci.edu/ml/datasets/Diabetes <br>
Kidney Disease : https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease

## MIMIC III Demo Dataset
We used these instructions to access the MIMIC III demo data: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/ <br>
Visit dataset: `SELECT  * FROM physionet-data.mimiciii_demo.admissions`

## RESULTS

TODO -- include table of results!
