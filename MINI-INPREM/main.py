#  -*- coding: utf-8 -*-
# @Time : 2019/11/14 下午5:19
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn

'''
EDITED BY: Nadia Wood (NW) and Diana Gonzalez Santillan (DGS)
NOTE: Edits made by use are marked with out initials at the top of each code block
'''

import os
import argparse
import torch
import torch.nn.functional as F
from doctor.model import Inprem
from Loss import UncertaintyLoss

''' IMPORTS by NW and DGS: '''
import pandas as pd
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
from datetime import datetime

# diagnoses bestsetting batch 32 lr 0.0005 l2 0.0001 drop 0.5 emb 256 starval 50 end val 65
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('diagnoses', 'heart'), # excluded 'diabetes', 'kidney'
                        help='Choose a task.', default='heart')
    parser.add_argument('--data_root', type=str, default='../datasets/',
                        help='The dataset root dir.')
    parser.add_argument('--fold', choices=(1, 2, 3, 4, 5), default=1, help='Choose a fold.')

    parser.add_argument('--use_cuda', action='store_true',
                        help='If use GPU.', default=False)
    parser.add_argument('--gpu_devices', type=str, default='0',
                        help='Choose devices ID for GPU.')
    parser.add_argument('--epochs', type=int, default=25, help='Setting epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini-batch size')
    parser.add_argument('--drop_rate', type=float, default=0.5,
                        help='The drop-out rate before each weight layer.')
    parser.add_argument('--optimizer', choices=('Adam', 'SGD', 'Adadelta'),
                        help='Choose the optimizer.', required=False, default='Adam')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='The learning rate for each step.')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Setting weight decay')

    parser.add_argument('--n_head', type=int, default=2,
                        help='The number of head of self-attention for the visit attention.')
    parser.add_argument('--n_depth', type=int, default=2,
                        help='The number of layers of self-attention for the visit attention.')

    parser.add_argument('--emb_dim', type=int, default=128,
                        help='The size of medical variable (or code) embedding.')

    parser.add_argument('--d_k', type=int, default=128,
                        help='The size of vector before self attention ')
    parser.add_argument('--d_v', type=int, default=128,
                        help='The size of vector before self attention ')
    parser.add_argument('--d_inner', type=int, default=128,
                        help='')
    parser.add_argument('--dvp', action='store_true', default=False,
                        help='Weather use position embedding.')
    parser.add_argument('--dp', action='store_true', default=False,
                        help='Weather use position embedding.')
    parser.add_argument('--ds', action='store_true', default=False, help='whether delete the sparse_max')
    parser.add_argument('--cap_uncertainty', action='store_true',
                        help='Weather capture uncertainty.', default=True)
    parser.add_argument('--monto_carlo_for_aleatoric', type=int, default=100,
                        help='The size of Monto Carlo Sample.')
    # Note: Previous monto_carlo_for_epistemic default was 200 (significantly increases epoch time)
    parser.add_argument('--monto_carlo_for_epistemic', type=int, default=10,
                        help='The size of Monto Carlo Sample.')
    parser.add_argument('--analysis_dir', type=str, default='../output_for_analysis_final/',
                        help='Set the analysis output dir')
    parser.add_argument('--write_performance', action='store_true', default=False,
                        help='Weather write performance result')
    parser.add_argument('--performance_dir', type=str, default='../metric_results/',
                        help='Set the performance dir')
    parser.add_argument('--save_model_dir', type=str, default='../saved_model/',
                        help='Set dir to save the model which has the best performance.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Choose the model dict to load for test or fine-tune.')
    parser.add_argument('--data_scale', default=1, type=float)
    return parser

def monto_calo_test(net, seq, mask, T):
    N = 3

    out, aleatoric = None, None
    outputs = []
    for i in range(T):
        seed = random.randint(0, 100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        out_instance = net(seq, mask)
        aleatoric = out_instance[:, N:]
        if i == 0:
            out = F.softmax(out_instance[:, :N], dim=1)
            outputs.append(F.softmax(out_instance[:, :N], dim=1).cpu().detach().numpy())
        else:
            out = out + F.softmax(out_instance[:, :N], dim=1)
            aleatoric = aleatoric + out_instance[:, N:]
            outputs.append(F.softmax(out_instance[:, :N], dim=1).cpu().detach().numpy())
    out = out / T
    aleatoric = aleatoric / T
    epistemic = -torch.sum(out * torch.log(out), dim=1)

    return out, aleatoric, epistemic, outputs

''' CLASS ADDED BY NW  '''
class InpremData(Dataset):
    max_visit = 0
    x = []
    mask = []
    y = []

    def __init__(self, visit_data, visit_mask, disease_status):
        self.x = visit_data
        self.mask = visit_mask
        self.y = disease_status
        self.max_visit = visit_data.shape[1]

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.mask[index], self.y[index])

'''
main METHOD EDITED BY NW AND DGS

NW:
- General dataset loading and overall data setup -- InpremData class
- wrangling of heart dataset to be usable with our code (created heart csv file)
- Epoch loop for selecting loss function, selecting optimizer, and training model
- Fixed shape and value mismatch with accuracy reporting
- Added monto_calo_test in the test/validation phases

DGS:
- Initial pseudocode for dataset loading and for training (later replaced by actual code)
- Logic to choose correct dataset based on task and split data into train test valid
- split code into functions to clearly show preprocessing, training, evaluation
- validation part (based on https://pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- testing part (based on https://docs.microsoft.com/en-us/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model)


    Diana notes:
    in the paper it is also mentioned that p_k = 0.5 , T_mc = 50 and T_test = 100
    not sure where these numbers will come in handy but adding a note just incase
    and also stacked multihead attention is stacked 2 times.
''' 
def main(opts):

    ''' setup GPUs if applicable'''
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_devices

    ''' preprocessing code '''
    train_set, valid_set, test_set, input_dim = preprocessing(opts.task, opts.data_root)

    ''' Define the model '''
    net = Inprem(opts.task, input_dim, 2, opts.emb_dim,
                 max(train_set.max_visit, valid_set.max_visit, test_set.max_visit),
                 opts.n_depth, opts.n_head, opts.d_k, opts.d_v, opts.d_inner,
                 opts.cap_uncertainty, opts.drop_rate, False, opts.dp, opts.dvp, opts.ds)
    
    model_path = opts.save_model_dir + 'pretrained.pt'

    ''' train and validate model '''
    training(opts, net, train_set, valid_set, model_path)
    ''' NOTE: at this point we will have saved the best pretrained model in model_path '''

    ''' test (saved) model '''
    evaluate(net, test_set, model_path)

    return

''' preprocessing code '''
def preprocessing(task, data_root):

    ''' determine which data to load'''
    DATA_PATH = data_root
    if(task == 'heart'):
        data_csv = 'HeartFinalDataset.csv'
    elif(task == 'diagnoses'):
        data_csv = 'diagnosisCode.csv'
    else:
        print('please select a valid task')
        return None

    ''' load the data '''
    # TODO - data wrangling for diagnoses dataset (below works for heart)
    load_data = pd.read_csv(DATA_PATH + data_csv).drop_duplicates().sort_values(by=['SUBJECT_ID', 'CHARTDATE'])
    codes = load_data['CPT_CD'].drop_duplicates()
    load_data_by_subject = load_data.groupby('SUBJECT_ID').count()
    max_visits = load_data_by_subject['CHARTDATE'].max()

    ''' setup the data '''
    data = torch.zeros( (load_data_by_subject.shape[0], max_visits, codes.shape[0]) )
    visit_mask = torch.zeros( (load_data_by_subject.shape[0], max_visits) )
    disease_status_by_subject = torch.zeros( (load_data_by_subject.shape[0], 1) )
    for index, entry in enumerate(load_data_by_subject.iterrows()):
        subject_id = entry[0];
        has_disease = False

        visit_nbr = 0
        for visit in load_data.iterrows():
            entry = visit[1]
            #print(visit)
            if entry['SUBJECT_ID'] == subject_id:
                #dt = entry['CHARTDATE']
                code = entry['CPT_CD']
                has_disease = entry['HAS_DIAG'] == 1
                for code_idx, code_record in enumerate(codes.iteritems()):
                    if code == code_record[1]:
                        data[index][visit_nbr][code_idx] = 1
                visit_mask[index][visit_nbr] = 1
                visit_nbr += 1

        if has_disease:
            disease_status_by_subject[index] = 1.0

    ''' split the data '''
    # Split subject IDs into 3 groups (75% train, 10% valid, 15% test)
    total = data.shape[0]
    seven_five = math.floor((75 * total) / 100)
    ten = math.floor((10 * total)/ 100)
    fifteen = total - seven_five - ten

    train_data, valid_data, test_data = torch.split(data, [seven_five, ten, fifteen])

    train_set = InpremData(train_data, visit_mask, disease_status_by_subject)
    valid_set = InpremData(valid_data, visit_mask, disease_status_by_subject)
    test_set  = InpremData(test_data , visit_mask, disease_status_by_subject)

    input_dim = codes.shape[0] # Number of distinct codes in the data set

    return (train_set, valid_set, test_set, input_dim)

''' training code '''
def training(opts, net, train_set, valid_set, model_path):
    '''Select loss function'''
    if opts.cap_uncertainty:
        criterion = UncertaintyLoss(opts.task, opts.monto_carlo_for_aleatoric, 2)
    else:
        criterion = CrossEntropy(opts.task)

    ''' select use of GPUs'''
    if opts.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    '''Select optimizer'''
    optimizer = None
    if (opts.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    elif (opts.optimizer == 'SGD'):
        optimizer = torch.optim.SGD(net.parameters(), lr=opts.lr, weight_decay=opts.weight_deca)
    elif (opts.optimizer == 'Adadelta'):
        optimizer = torch.optim.Adadelta(net.parameters(), lr=opts.lr, weight_decay=opts.weight_deca)
    else: # default = Adam
        optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        

    ''' get loaders and initial values'''
    train_loader = DataLoader(train_set, batch_size = 32)
    valid_loader = DataLoader(valid_set, batch_size = 32)
    best_vloss = 1_000_000 # just very large
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    ''' run epochs '''
    for epoch in range(0, opts.epochs): #default is 25 epochs,we can use --epochs option to change it
        print('\nEpoch ' + str(epoch+1) + ' of ' + str(opts.epochs))
        
        ''' train model '''
        print("TRAINING")
        net.train(True)
        train_loss = 0
        for (x, mask, y) in train_loader:

            optimizer.zero_grad()
            #out = net(x, mask)
            out, aleatoric, epistemic, outputs = monto_calo_test(net, x, mask, opts.monto_carlo_for_epistemic)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_loss = train_loss / len(train_loader)
        print('Training Loss =' + str(avg_loss))

        ''' validate model AND save best one '''
        print("VALIDATING")
        net.train(False)
        valid_loss = 0.0
        for (v_x, v_mask, v_y) in valid_loader:
            #voutputs = net(v_x, v_mask)
            voutputs, aleatoric, epistemic, outputs = monto_calo_test(net, v_x, v_mask, opts.monto_carlo_for_epistemic)
            vloss = criterion(voutputs, v_y)
            valid_loss += vloss

        avg_vloss = valid_loss / len(valid_loader)
        print('Validation Loss =' + str(avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = model_path
            # if there's already one we will replace it so we just save best one 
            torch.save(net.state_dict(), model_path)


''' evaluation code '''
def evaluate(net, test_set, model_path):
    print("\nTESTING")
    ''' test model '''
    test_loader = DataLoader(test_set, batch_size = 32)
    net.load_state_dict(torch.load(model_path)) 
    net.eval()
    total = 0
    running_accuracy = 0
    #y_size_shown = False
    for (x, mask, y) in test_loader: 
        y = y.flatten() != 0
        predicted_outputs = net(x,mask)
        _, predicted = torch.max(predicted_outputs, 1)
        predicted = predicted != 0
        total += y.size(0)

        #if not y_size_shown:
        #    y_size_shown = True
        #    print(y)
        #    print(predicted)
        #    print(y.size(0))
        #    print(y.shape)
        #    print((predicted == y))
        running_accuracy += (predicted == y).sum().item() 

    print("Accuracy: " + str(100 * running_accuracy / total))

''' main method for executable '''
if __name__ == '__main__':
    opts = args().parse_args()
    main(opts)
