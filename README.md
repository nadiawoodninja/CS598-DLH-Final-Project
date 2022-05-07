# CS598 DLH Final Project Spring 2022
> ### Project Team members
>  Nadia Wood (nadiaw2) and Diana Gonzalez Santillan (dianag4)


### Links 
---

Video Presentation Link => [Access here](https://www.youtube.com/watch?v=1m-L98yigJM) <br>
Silde Deck => [Access here ](https://github.com/nadiawoodninja/CS598DLHFinalProject/blob/main/CS598-DLH-Spring2022.pdf) <br>
Our Paper => [Access here](https://github.com/nadiawoodninja/CS598DLHFinalProject/blob/main/Reproducibility_Project_Instructions_for_CS598_DL4H_in_Spring_2022.pdf)

### Citations

---

The code in this repository is an adaptation of the original code sent to us by Xianli Zhang through email (xlbryant@stu.xjtu.edu.cn), downloadable from this Google Drive folder:
https://drive.google.com/file/d/1hfhM93zu_pc-SC2ppC6PTp5cFdfLjnsG/view?usp=sharing

Our work is based on a paper by Xianli Zhang et al. from Xi'an Jiaotong University, in Xi'an, China:

<b>Zhang, Xianli, et al.</b> “INPREM: An Interpretable and Trustworthy Predictive Model for Healthcare.” Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020, https://doi.org/10.1145/3394486.3403087

### Running the Code

---

#### Python version: Python3 
#### IDE: Visual Studio Code 
In order to run the code provided by the authors of the paper, we had to install the following dependencies. 
We used Visual Studio Code to import the project, and then installed the following required libraries:

```sh
sudo pip3 install torch torchvision
sudo pip3 install Cython
sudo pip3 install torchsparseattn
sudo pip3 install pandas
sudo pip3 install torchmetrics
sudo pip3 install torchsummary
sudo pip3 install tensorboard
```

Once the dependencies are installed you can run the preprocessing, training, and evaluation code by executing the follwing command:
```sh

python3 main.py --task=[TASK] --emb_dim=256 --d_k=256 --d_v=256 --d_inner=256
```

where `[TASK]` must be one of `diagnoses`, or `heart`. Note: Original code included diabetes and kidney disease tasks as well, but we have excluded that from our reproduction for simplicity.

Feel free to change any of the other hyperparameters to see the changes in the results of the model. Here is the full list of options available for you to run the model, along with their default values:

`--task`, choices=('diagnoses', 'heart'), default='heart', task to run
<br>
`--data_root`, type=str, default='../datasets/', dataset root directory
<br>
`--fold`, choices=(1, 2, 3, 4, 5), default=1, number of fold
<br>
`--use_cuda`, action='store_true', default=False, if use GPU
<br>
`--gpu_devices`, type=str, default='0', device IDs for GPU
<br>
`--epochs`, type=int, default=25, number of epochs
<br>
`--batch_size`, type=int, default=32
<br>
`--drop_rate`, type=float, default=0.5
<br>
`--optimizer`, choices=('Adam', 'SGD', 'Adadelta'), required=False, default='Adam'
<br>
`--lr`, type=float, default=5e-4, learning rate
<br>
`--weight_decay`, type=float, default=1e-4
<br>
`--n_head`, type=int, default=2, number of head of self-attention for the visit attention
<br>
`--n_depth`, type=int, default=2, number of layers of self-attention for the visit attention
<br>
`--emb_dim`, type=int, default=128, size of medical variable (or code) embedding.
<br>
`--d_k`, type=int, default=128, size of vector before self attention
<br>
`--d_v`, type=int, default=128, size of vector before self attention
<br>
`--d_inner`, type=int, default=128
<br>
`--dvp`, action='store_true', default=False, Weather use position embedding
<br>
`--dp`, action='store_true', default=False, Weather use position embedding
<br>
`--ds`, action='store_true', default=False, whether delete the sparse_max
<br>
`--cap_uncertainty`, action='store_true', Weather capture uncertainty, default=True
<br>
`--monto_carlo_for_aleatoric`, type=int, default=100, size of Monto Carlo Sample
<br>
`--monto_carlo_for_epistemic`, type=int, default=200, size of Monto Carlo Sample
<br>
`--analysis_dir`, type=str, default='../../output_for_analysis_final/', nalysis output dir
<br>
`--write_performance`, action='store_true', default=False, Weather write performance result
<br>
`--performance_dir`, type=str, default='../../metric_results/', performance dir
<br>
`--save_model_dir`, type=str, default='../../saved_model', set dir to save the model which has the best performance
<br>
`--resume`, type=str, default=None, Choose the model dict to load for test or fine-tune
<br>
`--data_scale`, default=1, type=float

NOTE: If you plan to use GPU computation, install CUDA: https://developer.nvidia.com/cuda-downloads and include the --use_cuda=True flag.

### Data Wrangling Phase of the Project 

---

#### MIMIC III Demo Dataset from Google's BigQuery

We used the publicly available MIMIC III dataset to acquire the diagnosis codes dataset, and the Heart Failure dataset needed for the project. Specifically, we followed these instructions to access the MIMIC III demo data: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/ <br>

Then, we used the following queries to get the specific datasets we needed for the project:
<br>


##### To generate mimiciiiDemoData.csv we used the query below 

`SELECT * FROM 'physionet-data.mimiciii_demo.admissions`

##### To generate diagnosisCode.csv we used the query below

`SELECT * FROM 'physionet-data.mimiciii_demo.diagnoses_icd` 


##### To generate HeartFinalDataset.csv we used the query below

```
SELECT c.SUBJECT_ID, c.CHARTDATE, c.CPT_CD, 
CASE WHEN d.ICD9_CODE IS null THEN 0 ELSE 1 END AS HAS_DIAG
FROM 'physionet-data.mimiciii_demo.cptevents' c
LEFT JOIN 'physionet-data.mimiciii_demo.diagnoses_icd' d
ON c.SUBJECT_ID=d.SUBJECT_ID and d.ICD9_CODE='42731'
WHERE c.CHARTDATE IS NOT null
ORDER BY c.SUBJECT_ID, c.CHARTDATE
```


### Data Profiling and Stats - Understanding the Data 

---

**Mimiciii Demo Admissions Data Stats**

mimiciiiDemoData.csv contains demo data from the **table** mimiciii_demo.admissions. Using panda-profiling we were quickly able to get a sense of the data and statistics about this dataset. A detailed report is avaiable here=> [Data Profiling Report](https://htmlpreview.github.io/?https://github.com/nadiawoodninja/CS598DLHFinalProject/blob/main/mimiciiiDemoDataStats.html)

![image](https://user-images.githubusercontent.com/50491061/166164466-87322015-9591-4379-b3e9-3e2db6b08443.png)


**Diagnosis Code Data Stats**

A detailed report is avaiable here=> [Data Profiling Report](https://htmlpreview.github.io/?https://github.com/nadiawoodninja/CS598DLHFinalProject/blob/main/DiagnosisCodeDataStats.html)

![image](https://user-images.githubusercontent.com/50491061/166164810-646bccdc-4d8d-4d86-9a2f-85146439679f.png)

**Heart Disease Data Stats**

A detailed report is avaiable here=> [Data Profiling Report](https://htmlpreview.github.io/?https://github.com/nadiawoodninja/CS598DLHFinalProject/blob/main/HeartDiseaseDataStats.html)

![image](https://user-images.githubusercontent.com/50491061/166164871-2d7a123f-d3de-4ec9-8b37-1073bdd55262.png)

Alerts for the dataset that was used to train the model. High cerdinality and duplicate rows were found in the dataset.

![image](https://user-images.githubusercontent.com/50491061/166165217-8806d936-589b-4f29-90b5-1ed5f1cc381b.png)


### Results

---
**Data for training model:** We split subject IDs into 3 groups (75% train, 10% valid, 15% test)

Model analysis results are in the folder=>[Results](https://github.com/nadiawoodninja/CS598DLHFinalProject/tree/main/output_for_analysis_final)

|Epocs|	Batch Size |	Drop Rate |	Learning Rate |	Weight Decay | Accuracy |	F1 Score |
|-----|------------|------------|---------------|--------------|----------|----------|						
|5|	32|	0.5|	0.0005|	0.0001|	43%	|0.33|
|10|	32|	0.5|	0.0005|	0.0001|	86%|	0.89|
|15|	32|	0.5|	0.0005|	0.0001|	71%	|0.75|
|20|	32|	0.5|	0.0005|	0.0001|	57%	|0.73|
|25|	32|	0.5|	0.0005|	0.0001|	57%	|0.67|
|30|	32|	0.5|	0.0005|	0.0001|	57%	|0.40|


#### Model Visualization
```sh
Inprem(
  (embedding): Linear(in_features=2, out_features=128, bias=False)
  (position_embedding): Embedding(35, 128)
  (encoder): Encoder(
    (layer_stack): ModuleList(
      (0): EncoderLayer(
        (slf_attn): MultiHeadAttention(
          (w_qs): Linear(in_features=128, out_features=256, bias=True)
          (w_ks): Linear(in_features=128, out_features=256, bias=True)
          (w_vs): Linear(in_features=128, out_features=256, bias=True)
          (attention): ScaledDotProductAttention(
            (dropout): Dropout(p=0.5, inplace=False)
            (softmax): Softmax(dim=2)
          )
          (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (fc): Linear(in_features=256, out_features=128, bias=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (pos_ffn): PositionwiseFeedForward(
          (w_1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (w_2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (1): EncoderLayer(
        (slf_attn): MultiHeadAttention(
          (w_qs): Linear(in_features=128, out_features=256, bias=True)
          (w_ks): Linear(in_features=128, out_features=256, bias=True)
          (w_vs): Linear(in_features=128, out_features=256, bias=True)
          (attention): ScaledDotProductAttention(
            (dropout): Dropout(p=0.5, inplace=False)
            (softmax): Softmax(dim=2)
          )
          (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (fc): Linear(in_features=256, out_features=128, bias=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (pos_ffn): PositionwiseFeedForward(
          (w_1): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (w_2): Conv1d(128, 128, kernel_size=(1,), stride=(1,))
          (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
    )
  )
  (w_alpha_1): Linear(in_features=128, out_features=1, bias=True)
  (w_alpha_2): Linear(in_features=128, out_features=1, bias=True)
  (w_beta): Linear(in_features=128, out_features=128, bias=True)
  (variance): Linear(in_features=128, out_features=1, bias=True)
  (predict): Linear(in_features=128, out_features=2, bias=True)
  (dropout): Dropout(p=0.5, inplace=False)
  (sparsemax): Sparsemax()
)
```

### Appendix: Communication with Authors

---

<img width="1133" alt="1" src="https://user-images.githubusercontent.com/80500914/166115836-37eba716-438c-4a97-9e4e-ab3bb3aafb24.png">
<img width="1041" alt="2" src="https://user-images.githubusercontent.com/80500914/166115851-01c83462-dda4-4b9e-b26f-e6b125b5beb9.png">
<img width="1057" alt="3" src="https://user-images.githubusercontent.com/80500914/166115852-ab31edfa-a7fa-4403-8abb-7517e483ffda.png">
<img width="810" alt="4" src="https://user-images.githubusercontent.com/80500914/166115846-7349c179-99ba-4ad8-8e09-e4ac9f83f01d.png">
<img width="336" alt="5" src="https://user-images.githubusercontent.com/80500914/166115847-21ba704b-8aba-414f-9e5d-8b4d0317e738.png">
<img width="1077" alt="6" src="https://user-images.githubusercontent.com/80500914/166115849-2b3a2152-a0cf-4ea1-9044-57da739f9751.png">
<img width="1070" alt="7" src="https://user-images.githubusercontent.com/80500914/166115844-34819b3b-9080-4b2c-b25d-41aa6da485ea.png">
<img width="639" alt="8" src="https://user-images.githubusercontent.com/80500914/166115842-a2845764-4c16-40ba-a5a8-405b7181020d.png">
<img width="888" alt="9" src="https://user-images.githubusercontent.com/80500914/166115853-803959cb-c1ad-41ed-83d0-069562c142d1.png">

Note: We did not choose to contact Professor Sum as we were able to find a way to wrangle the data they way we needed for it to work with the model.
