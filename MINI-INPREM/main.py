#  -*- coding: utf-8 -*-
# @Time : 2019/11/14 下午5:19
# @Author : Xianli Zhang
# @Email : xlbryant@stu.xjtu.edu.cn
import os
import argparse
import torch
import torch.nn.functional as F
from doctor.model import Inprem
from Loss import UncertaintyLoss
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# diagnoses bestsetting batch 32 lr 0.0005 l2 0.0001 drop 0.5 emb 256 starval 50 end val 65
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('diagnoses', 'heart', 'diabetes', 'kidney'),
                        help='Choose a task.', default='heart')
    parser.add_argument('--data_root', type=str, default='../../datasets/',
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
                        help='Choose the optimizer.', required=False)
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
    parser.add_argument('--monto_carlo_for_epistemic', type=int, default=200,
                        help='The size of Monto Carlo Sample.')
    parser.add_argument('--analysis_dir', type=str, default='../../output_for_analysis_final/',
                        help='Set the analysis output dir')
    parser.add_argument('--write_performance', action='store_true', default=False,
                        help='Weather write performance result')
    parser.add_argument('--performance_dir', type=str, default='../../metric_results/',
                        help='Set the performance dir')
    parser.add_argument('--save_model_dir', type=str, default='../../saved_model',
                        help='Set dir to save the model which has the best performance.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Choose the model dict to load for test or fine-tune.')
    parser.add_argument('--data_scale', default=1, type=float)
    return parser


def monto_calo_test(net, seq, mask, T):
    out, aleatoric = None, None
    outputs = []
    for i in range(T):
        seed = random.randint(0, 100)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        out_instance = net(seq, mask)
        aleatoric = out_instance[:, 2:]
        if i == 0:
            out = F.softmax(out_instance[:, :2], dim=1)
            outputs.append(F.softmax(out_instance[:, :2], dim=1).cpu().detach().numpy())
        else:
            out = out + F.softmax(out_instance[:, :2], dim=1)
            aleatoric = aleatoric + out_instance[:, 2:]
            outputs.append(F.softmax(out_instance[:, :2], dim=1).cpu().detach().numpy())
    out = out / T
    aleatoric = aleatoric / T
    epistemic = -torch.sum(out * torch.log(out), dim=1)

    return out, aleatoric, epistemic, outputs

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

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_devices

    # Nadia - Load dataset
    DATA_PATH = "../datasets/"
    
    heart_data = pd.read_csv(DATA_PATH + 'HeartFinalDataset.csv').drop_duplicates().sort_values(by=['SUBJECT_ID', 'CHARTDATE'])

    codes = heart_data['CPT_CD'].drop_duplicates()

    input_dim = codes.shape[0] # Number of distinct codes in the data set

    heart_data_by_subject = heart_data.groupby('SUBJECT_ID').count()
    max_visits = heart_data_by_subject['CHARTDATE'].max()

    data = torch.zeros( (heart_data_by_subject.shape[0], max_visits, codes.shape[0]) )
    visit_mask = torch.zeros( (heart_data_by_subject.shape[0], max_visits) )
    disease_status_by_subject = torch.zeros( (heart_data_by_subject.shape[0], 1) )
    for index, entry in enumerate(heart_data_by_subject.iterrows()):
        subject_id = entry[0];
        has_disease = False

        visit_nbr = 0
        for visit in heart_data.iterrows():
            entry = visit[1]
            #print(visit)
            if entry['SUBJECT_ID'] == subject_id:
                #dt = entry['CHARTDATE']
                code = entry['CPT_CD']
                has_disease = entry['HAS_DIAG'] == 1
                #print(code)
                for code_idx, code_record in enumerate(codes.iteritems()):
                    #print(code_record)
                    if code == code_record[1]:
                        data[index][visit_nbr][code_idx] = 1
                visit_mask[index][visit_nbr] = 1
                visit_nbr += 1

        if has_disease:
            disease_status_by_subject[index] = 1.0
        #print(data[index])
        #print(visit_mask[index])
        #print(disease_status_by_subject[index])

    # data = torch.zeros( (heart_data_by_subject.shape[0], codes.shape[0], max_visits) )
    # data = torch.zeros( (heart_data_by_subject.shape[0], 3, max_visits) )
    # data = torch.ones( (heart_data_by_subject.shape[0], 34, 2) )
    # data = torch.ones( (heart_data_by_subject.shape[0], max_visits, codes.shape[0]) )

    # disease_status_by_subject = torch.zeros( (heart_data_by_subject.shape[0], 1) )

    train_set = InpremData(data, visit_mask, disease_status_by_subject)
    valid_set = InpremData(data, visit_mask, disease_status_by_subject)
    test_set = InpremData(data, visit_mask, disease_status_by_subject)

    # print(train_set.max_visit)
    # train_set.max_visit = max_visits
    # valid_set.max_visit = max_visits
    # test_set.max_visit = max_visits

    # Split subject IDs into 3 groups (75% train, 10% valid, 15% test)

    # input_dim will be fed to torch.nn.Linear function (in_features parameter)
    # Maybe number of diagnosis codes?
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

    ''' Diana notes LOADING DATA:
    0. 
    Command to run the code (change 'task' option according to what we want):

    python3 main.py task=['diagnoses', 'heart', 'diabetes', 'kidney']
                    emb_dim=256 (all these 256s are 128 by default but paper states they used 256, but if it is too slow we can try the defaults)
                    d_k=256
                    d_v=256
                    d_inner=256
                    use_cuda=False (unless we are using a GPU, in that case need to include gpu_devices too)
                    optimizer=Adam

    I only included all the opts that I know the values for from the paper... 
    (we can leave defaults for: epochs, batch_size, drop_rate, lr, weight_decay,
     n_head, n_depth, ... and all the other ones that appear in args() fcn above)

    1.
    I think we can use the --data_root option when running main.py and set it to
    --data_root=../datasets
    (OR we can just change the default in the args() fcn to be ../datasets)

    then opts.data_root will be the path to the datasets...
    we can use that to file.open() the necessary dataset and somehow parse it...
    to choose which file to file.open() we can check opts.task which will be one of
    'diagnoses', 'heart', 'diabetes', or 'kidney' and depending on the task open the
    correct dataset then...

    2. 
    Then we need to split the data into train, valid, and test with a ratio of
    75:10:15 (thats the ratio they used in the paper), we should put the data in
    train_set.data (.data makes sense to me but I think this can be whatever)
    valid_set.data
    test_set.data

    3.
    Also for each we need to figure out the max visit and save it in
    train_set.max_visit
    valid_set.max_visit
    test_set.max_visit
    respectively

    4. we need to figure out what input_dim in the inprem definition should be... 256??

    once we have steps 1-4 above I think the net model below should be defined like we
    want it, next would be training etc, I have added notes about that under that TO DO
    '''

    # # 2x34 and 2x128
    # print(input_dim) # Based on data
    # print(max(train_set.max_visit, valid_set.max_visit, test_set.max_visit)) # Based on data
    # # Note: out_dim = 2
    # print(opts.emb_dim) # 128
    # print(opts.n_depth) # 2
    # print(opts.n_head) # 2
    # print(opts.d_k) # 128
    # print(opts.d_v) # 128
    # print('---')

    '''Define the model.'''
    net = Inprem(opts.task, input_dim, 2, opts.emb_dim,
                 max(train_set.max_visit, valid_set.max_visit, test_set.max_visit),
                 opts.n_depth, opts.n_head, opts.d_k, opts.d_v, opts.d_inner,
                 opts.cap_uncertainty, opts.drop_rate, False, opts.dp, opts.dvp, opts.ds)

    '''Select loss function'''
    if opts.cap_uncertainty:
        criterion = UncertaintyLoss(opts.task, opts.monto_carlo_for_aleatoric, 2)
    else:
        criterion = CrossEntropy(opts.task)

    if opts.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    '''Select optimizer'''
    optimizer = torch.optim.Adam(net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)


    #TODO: Training, validating, and testing.

    train_loader = DataLoader(train_set, batch_size = 32)

    # Train
    net.train()
    for epoch in range(0, opts.epochs):
        print('epoch = ' + str(epoch))
        for (x, mask, y) in train_loader:
            train_loss = 0

            optimizer.zero_grad()
            out = net(x, mask)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    # Validate
    # Not required if hyper parameters have already been determined?

    # Test
    net.eval()
    return
    '''
    Diana notes: I wrote a skeleton for the epochs based on what I remember from the HWs...
    '''
    # TRAINING
    for epoch in range(0,opts.epochs): #default is 25 epochs,we can use --epochs option to change it
        '''
        Diana notes:
        the forward method for Inprem is forward(self,seq,mask)
        so I guess seq will be a sequence of something? visits? from the data
        not sure about mask, there's no mask mentioned in the paper :(

        in the paper it is also mentioned that p_k = 0.5 , T_mc = 50 and T_test = 100
        not sure where these numbers will come in handy but adding a note just incase
        and also stacked multihead attention is stacked 2 times.
        '''
        print('this is an epoch')
        net.train()
        train_loss = 0
        #figure out what the optimizer is
        #figure out what the loss criterion is
        #for sequence in train_set.data: # this might be different
            # figure out what the mask is
            # zero out gradient: optimizer.zero_grad()
            # send stuff to forward -- y_hat = net(sequence,mask) or something
            # use loss criterion and do backward pass
            # optimizer.step()
            # train_loss += loss.item()
        # train_loss = train_loss / len(train_loader)
        valid_or_test = False
        if valid_or_test: # not sure if this should be here or where
            # In valid and test phase, you should use the monto_calo_test()
            print('validation or test')
            # monto_calo_test(net, input, mask, opts.monto_carlo_for_epistemic)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1, train_loss))

    # VALIDATING

    # TESTING

if __name__ == '__main__':
    opts = args().parse_args()
    main(opts)