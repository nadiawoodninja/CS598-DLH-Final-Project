# CS598 DLH Final Project Spring 2022
## Nadia Wood (nadiaw2) and Diana Gonzalez Santillan (dianag4)

### Citations

The code in this repository is an adaptation of the original code sent to us by Xianli Zhang through email (xlbryant@stu.xjtu.edu.cn), downloadable from this Google Drive folder:
https://drive.google.com/file/d/1hfhM93zu_pc-SC2ppC6PTp5cFdfLjnsG/view?usp=sharing

Our work is based on a paper by Xianli Zhang et al. from Xi'an Jiaotong University, in Xi'an, China:

<b>Zhang, Xianli, et al.</b> “INPREM: An Interpretable and Trustworthy Predictive Model for Healthcare.” Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2020, https://doi.org/10.1145/3394486.3403087

### Running the code

#### Python version: Python3 
#### IDE: Visual Studio Code 
In order to run the code provided by the authors of the paper, we had to install the following dependencies. 
We used Visual Studio Code to import the project, and then intalled the following required libraries:
<br>
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

`python3 main.py --task=[TASK] --emb_dim=256 --d_k=256 --d_v=256 --d_inner=256`
<br>

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

### Open Datasets Used

#### MIMIC III Demo Dataset

We used the publicly available MIMIC III dataset to acquire the diagnosis codes dataset, and the Heart Failure dataset needed for the project. Specifically, we followed these instructions to access the MIMIC III demo data: https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/ <br>

Then, we used the following queries to get the specific datasets we needed for the project:
<br>
##### HeartFinalDataset.csv Query

```
SELECT c.SUBJECT_ID, c.CHARTDATE, c.CPT_CD, <br>
CASE WHEN d.ICD9_CODE IS null THEN 0 ELSE 1 END AS HAS_DIAG
FROM 'physionet-data.mimiciii_demo.cptevents' c
LEFT JOIN 'physionet-data.mimiciii_demo.diagnoses_icd' d
ON c.SUBJECT_ID=d.SUBJECT_ID and d.ICD9_CODE='42731'
WHERE c.CHARTDATE IS NOT null
ORDER BY c.SUBJECT_ID, c.CHARTDATE
```

##### diagnosisCode.csv Query
`SELECT * FROM 'physionet-data.mimiciii_demo.diagnoses_icd'` 

##### mimiciiiDemoData.csv Query
`SELECT * FROM 'physionet-data.mimiciii_demo.admissions'`

### RESULTS

TODO -- include table of results!

### APPENDIX: Communication With Authors

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
