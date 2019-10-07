import pandas as pd
import os
import numpy as np
import pickle
import numpy.ma as ma

import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

import random
from math import floor
import copy 

# from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.metrics import roc_auc_score, average_precision_score
import seaborn as sb

import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

################# Options ##################################################
############################################################################
parser = argparse.ArgumentParser(description='Binary classification for cancer',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, choices=['brca', 'ov', 'prad', 'lung', 'lusc', 'breast1', 'breast2'],
                    help='Choose the dataset.')
parser.add_argument('--data_dir', type=str, choices=['brca_ov_prad', 'lung_cancer','two_breast_cancer'],
                    help='Choose the data directory.')

args = parser.parse_args()


data_dir = args.data_dir
dataset_name = args.dataset

data_file = dataset_name + '_data.pkl'
label_file = dataset_name + '_label.pkl'

data_path = os.path.join(os.getcwd(),data_dir,data_file)
label_path = os.path.join(os.getcwd(),data_dir,label_file)

class Cancer_Dataset(Dataset):
    """
    The dataset wrapping for drug dataset. Data will only be
    loaded into memory through the dataloader function.
    
    Args:
        input_file (str): csv file path store the data of input data x.
        target_file (str): csv file path store the data of label data y.
        transform (callable, optional): Optional transform to be applied on a sample.
        split (callable, optional): Optional dataset split to be applied on dataset.
        
    Returns:
        
    """
    def __init__(self, input_file, target_file, transform=None):
        
        self.input_file = input_file
        self.target_file = target_file
        self.transform = transform # data transform
        
        self.input_data = np.transpose(pd.read_pickle(input_file))
        self.target_data = np.transpose(pd.read_pickle(target_file))
        
        # change the data precision from float64 to float32
        self.input_data = self.input_data.astype(np.float32, copy=False)
        self.target_data = self.target_data.astype(np.float32, copy=False)
        
        # the following reorder function perform the feature selection
        self.__reorder__()
        
        # store the index for the dataset
        # note that input_data is in the shape of (rna_index, patient_index)
        # while, target_data is in the shape of (patient_index, 1)
        self.input_sample = self.input_data.shape[1]
        self.input_feature = self.input_data.shape[0]
        self.target_sample = self.target_data.__len__()

        # check whether the sample names are match
        assert (self.input_sample == self.target_sample), "number of samples"\
            "not match between two sets"

        
    def __reorder__(self, non_test_idx = None):
        
        if non_test_idx is None:
            dummy_mat = np.append(self.input_data, np.expand_dims(self.target_data, axis=0), axis=0)
        else:
            dummy_mat = np.append(self.input_data[:,non_test_idx],
                                 np.expand_dims(self.target_data[non_test_idx], axis=0), axis = 0)
        
        C_mat = np.corrcoef(dummy_mat)
        C_shape = C_mat.shape[0]
        corr_list = C_mat[C_shape-1, :C_shape-1]
        
        # deal with the NaN when calculate the Pearson correlation coefficient
        mask = np.isnan(corr_list)
        corr_list[mask] = 0 # make the corr_list in NaN to be zero, thus can be sorted descendingly
        #store the corr_list
        self.corr_list = corr_list
        
        corr_index = corr_list.argsort()[::-1]
        # reorder the data w.r.t corr_index
        self.input_data = self.input_data[list(corr_index), :]

        return corr_list[list(corr_index)]
        
    
    # Override to give Pytorch access to data on the dataset
    def __getitem__(self, idx):
        
        sample = {'input_data': self.input_data[:,idx],
                  'target_data': np.array([self.target_data[idx]])}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample['input_data'], sample['target_data'] 
        
    # Override to give Pytorch the size of dataset
    def __len__(self):
        return self.input_sample
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        return {'input_data': torch.from_numpy(sample['input_data']),
                'target_data': torch.from_numpy(sample['target_data'])}
    

def data_split(dataset,
               split_ratio=[0.6, 0.2, 0.2],
               shuffle=False,
               manual_seed=None):

    length = dataset.__len__()
    indices = list(range(1, length))

    assert (sum(split_ratio) == 1), "Partial dataset is not used"

    if manual_seed is None:
        manual_seed = random.randint(1, 10000)

    if shuffle == True:
        random.seed(manual_seed)
        random.shuffle(indices)

    breakpoint_train = floor(split_ratio[0] * length)
    breakpoint_val = floor(split_ratio[1] * length)

    idx_train = indices[:breakpoint_train]
    idx_val = indices[breakpoint_train:breakpoint_train + breakpoint_val]
    idx_test = indices[breakpoint_train + breakpoint_val:]

    return idx_train, idx_val, idx_test


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model,
          criterion,
          optimizer,
          train_loader,
          epoch,
          reduction='avg',
          rank=50,
          print_log=True):

    losses = AverageMeter()

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        output = model(data[:, 0:rank])
        loss = criterion(output, target)

        losses.update(loss.data.item(), data.size(0))

        loss.backward()
        optimizer.step()

        if print_log:
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

    if reduction is 'sum':
        return losses.sum
    elif reduction is 'avg':
        return losses.avg


def validate(model,
             criterion,
             val_loader,
             epoch,
             reduction='avg',
             rank=50,
             print_log=True):

    losses = AverageMeter()
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = data.cuda()
            target = target.cuda()

            output = model(data[:, 0:rank])

            pred = classification(output)
            roc_auc = AUC_eval(pred, target, opt='roc')
            prc_auc = AUC_eval(pred, target, opt='prc')

            #             output_var = output.std()
            loss = criterion(output, target)
            losses.update(loss.data.item(), data.size(0))

    if print_log:
        print('Val Epoch:{} \t Loss: {:.6f} \t AUC {:.3f} \t AUPRC {:.3f}'.format(
            epoch, losses.avg, roc_auc, prc_auc))

    if reduction is 'sum':
        return losses.sum, roc_auc, prc_auc
    elif reduction is 'avg':
        return losses.avg, roc_auc, prc_auc


def classification(input):
    output = torch.sigmoid(input)
    return (torch.sign(output - 0.5) + 1) * 0.5


def AUC_eval(pred, target, opt='roc'):
    # ensure input args are in numpy format
    pred_np = copy.deepcopy(pred).cpu().numpy()
    target_np = copy.deepcopy(target).cpu().numpy()
    if opt == 'roc':
        # fpr, tpr, _ = roc_curve(pred_np, target_np)
        # output_auc = auc(fpr, tpr)
        output_auc = roc_auc_score(target_np, pred_np)
    elif opt == 'prc':
        # precision, recall, _ = precision_recall_curve(pred_np, target_np)
        # output_auc = auc(recall, precision)
        output_auc = average_precision_score(target_np, pred_np)
    return output_auc


def Pearson_loss(pred, target):
    v_pred = pred - torch.mean(pred)
    v_target = target - torch.mean(target)
    corr_coe = torch.sum(v_pred * v_target) / (
        torch.sqrt(torch.sum(v_pred**2)) * torch.sqrt(torch.sum(v_target**2)))
    # return torch.abs(corr_coe)
    return corr_coe


def corr_coef_eval(model, val_loader, trial_idx=0, rank=50, print_log=True):

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = torch.autograd.Variable(data).cuda()
            target = torch.autograd.Variable(target).cuda()

            output = model(data[0:rank, :])
            output_var = output.std()
            corr_coef = Pearson_loss(output, target)

        if print_log:
            print(
                'eval trial:{} \t Correlation Coefficient: {:.4f} \t Output Std {:.3f}'
                .format(trial_idx, corr_coef, output_var))

    return corr_coef


# Linear regression model
class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, 128, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.hidden2 = nn.Linear(128, 64, bias=False)
        self.bn2 = nn.BatchNorm1d(64)

        self.hidden3 = nn.Linear(64, 32, bias=False)
        self.bn3 = nn.BatchNorm1d(32)

        self.predict = nn.Linear(32, n_output, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight,
                                     mode='fan_in',
                                     nonlinearity='relu')
                #                 init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.hidden1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.hidden2(out)
        out = self.bn2(out)
        out = torch.relu(out)

        out = self.hidden3(out)
        out = self.bn3(out)
        out = torch.relu(out)

        out = self.predict(out)

        return out


# Hyper-parameters
N_input = 1000  # rank
# th_corr_list = [0.7, 0.6, 0.5]
th_corr_list = [1]
# th_corr_coe = 0.5
# N_input = np.greater(np.abs(data_set.corr_list),th_corr_coe).sum()
N_target = 1

LR = 0.1  # learning rate
W_decay = 0.001
N_EPOCHS = 100
print_log = print
N_trial = 50

dataset_transform = transforms.Compose([ToTensor()])
data_set = Cancer_Dataset(data_path, label_path, dataset_transform)

for th_corr_coe in th_corr_list:

    auc_log = []
    auprc_log = []

    for idx_trial in tqdm(range(N_trial)):

        # Criterion = nn.CrossEntropyLoss()
        Criterion = nn.BCEWithLogitsLoss().cuda()

        # creating a train/validation/test data split
        train_idx, val_idx, test_idx = data_split(data_set,
                                                  split_ratio=[0.6, 0.2, 0.2],
                                                  shuffle=True,
                                                  manual_seed=None)

        # NOTE: reorder the data based on the index
        data_set.__reorder__(train_idx + val_idx)

        # N_input = np.greater(np.abs(data_set.corr_list),th_corr_coe).sum()

        # Initialize the model and load into GPU
        model = Net(N_input, N_target).cuda()
        Optimizer = torch.optim.SGD(model.parameters(),
                                    lr=LR,
                                    momentum=0.9,
                                    nesterov=True,
                                    weight_decay=W_decay)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(Optimizer,
                                                           0.99,
                                                           last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(Optimizer, milestones=[80,160], gamma=0.1)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # create dataloaders from the same dataset but based on the different indices
        # sampler is mutually exclusive with shuffle, so here shuffle should be False
        train_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=train_idx.__len__(),
            sampler=train_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        val_loader = torch.utils.data.DataLoader(data_set,
                                                 batch_size=val_idx.__len__(),
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=4,
                                                 pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=test_idx.__len__(),
            sampler=test_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        loss_log = []
        best_val_auc = None
        best_val_auc_loss = None
        best_val_auprc = None
        best_val_auprc_loss = None

        for epoch in range(N_EPOCHS):
            scheduler.step()  # update the learning rate with scheduler\
            train_loss = train(model,
                               Criterion,
                               Optimizer,
                               train_loader,
                               epoch,
                               rank=N_input,
                               print_log=print_log)
            val_loss, val_auc, val_auprc = validate(model,
                                                    Criterion,
                                                    val_loader,
                                                    epoch,
                                                    reduction='avg',
                                                    rank=N_input,
                                                    print_log=print_log)

            # solve the nan problem for AUC
            if np.isnan(val_auc):
                val_auc = 0.5

            # solve the nan problem for AUPRC
            if np.isnan(val_auprc):
                val_auprc = 0

            # get the best model depends on AUC on validation data
            if best_val_auc is None:
                is_best_auc = True
                best_val_auc = val_auc
                best_val_auc_loss = val_loss
            else:
                is_best_auc = (val_auc > best_val_auc) or (
                    (val_loss < best_val_auc_loss) and
                    (val_auc == best_val_auc))
                best_val_auc = max(val_auc, best_val_auc)
                best_val_loss = min(val_loss, best_val_auc_loss)

            if is_best_auc:
                model_best_auc = copy.deepcopy(model)

            # get the best model depends on AUPRC on validation data
            if best_val_auprc is None:
                is_best_auprc = True
                best_val_auprc = val_auprc
                best_val_auprc_loss = val_loss
            else:
                is_best_auprc = (val_auprc > best_val_auprc) or (
                    (val_loss < best_val_auprc_loss) and
                    (val_auprc == best_val_auprc))
                best_val_auprc = max(val_auprc, best_val_auprc)
                best_val_auprc_loss = min(val_loss, best_val_auprc_loss)

            if is_best_auprc:
                model_best_auprc = copy.deepcopy(model)

            # loss_log += [[epoch, train_loss, val_loss, best_val_auc]]

        # perform the test with best model
        test_loss, test_auc, _ = validate(model_best_auc,
                               Criterion,
                               test_loader,
                               epoch,
                               reduction='avg',
                               rank=N_input,
                               print_log=print_log)

        auc_log += [[idx_trial, test_auc]]

        test_loss, _, test_auprc = validate(model_best_auprc,
                               Criterion,
                               test_loader,
                               epoch,
                               reduction='avg',
                               rank=N_input,
                               print_log=print_log)

        auprc_log += [[idx_trial, test_auprc]]

    ############## save results ###########
    ## auc
    trials = [x[0] for x in auc_log]
    tmp_auc_log = [x[1].item() for x in auc_log]
    tmp_a = np.asarray(tmp_auc_log, dtype=np.float32)
    np.savetxt('./save/' + dataset_name + '_' + str(N_input) + '_' +
               str(th_corr_coe) + '_auc.csv',
               auc_log,
               delimiter=",")

    fig, ax = plt.subplots()
    ax1 = plt.subplot(211)
    ax1.plot(trials, tmp_auc_log, 'ro')  #label='AUC'
    ax1.legend()
    ax1.set_xlabel('trial index')
    ax1.set_ylabel('AUC')

    ax2 = plt.subplot(212)
    mu = np.nanmean(tmp_a)
    median = np.nanmedian(tmp_a)
    sigma = np.nanstd(tmp_a)
    textstr = '\n'.join(
        (r'$\mu=%.2f$' % (mu, ), r'$\mathrm{median}=%.2f$' % (median, ),
         r'$\sigma=%.2f$' % (sigma, )))

    ax2.hist(tmp_a, 10)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax2.text(0.8,
             0.3,
             textstr,
             transform=ax.transAxes,
             fontsize=11,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=props)

    fig.savefig('./save/' + dataset_name + '_' + str(N_input) + '_' +
                str(th_corr_coe) + '_auc.png',
                bbox_inches='tight')


    ## auprc
    trials = [x[0] for x in auprc_log]
    tmp_auc_log = [x[1].item() for x in auprc_log]
    tmp_a = np.asarray(tmp_auc_log, dtype=np.float32)
    np.savetxt('./save/' + dataset_name + '_' + str(N_input) + '_' +
               str(th_corr_coe) + '_auprc.csv',
               auprc_log,
               delimiter=",")

    fig, ax = plt.subplots()
    ax1 = plt.subplot(211)
    ax1.plot(trials, tmp_auc_log, 'ro')  #label='AUC'
    ax1.legend()
    ax1.set_xlabel('trial index')
    ax1.set_ylabel('AUPRC')

    ax2 = plt.subplot(212)
    mu = np.nanmean(tmp_a)
    median = np.nanmedian(tmp_a)
    sigma = np.nanstd(tmp_a)
    textstr = '\n'.join(
        (r'$\mu=%.2f$' % (mu, ), r'$\mathrm{median}=%.2f$' % (median, ),
         r'$\sigma=%.2f$' % (sigma, )))

    ax2.hist(tmp_a, 10)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax2.text(0.8,
             0.3,
             textstr,
             transform=ax.transAxes,
             fontsize=11,
             horizontalalignment='center',
             verticalalignment='center',
             bbox=props)

    fig.savefig('./save/' + dataset_name + '_' + str(N_input) + '_' +
                str(th_corr_coe) + '_auprc.png',
                bbox_inches='tight')
