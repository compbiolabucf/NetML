# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import pickle
from sklearn.metrics import roc_curve, auc, average_precision_score
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

def read_data(names):
    data_1 = np.array(pickle.load(open(names[0], 'rb'))).T
    label_1 = np.array(pickle.load(open(names[1], 'rb')).flatten())
    data_2 = np.array(pickle.load(open(names[2], 'rb'))).T
    label_2 = np.array(pickle.load(open(names[3], 'rb')).flatten())
    data_3 = np.array(pickle.load(open(names[4], 'rb'))).T
    label_3 = np.array(pickle.load(open(names[5], 'rb')).flatten())
    print(data_1.shape,data_2.shape,data_3.shape)
    return data_1, data_2, data_3, label_1, label_2, label_3

def cross_validation(data_1_all, data_2_all, data_3_all, label_1_all,
                     label_2_all, label_3_all, test_number):
    ### get the test part
    data1_train, data1_test, label1_train, label1_test = train_test_split(
        data_1_all, label_1_all, test_size=test_number)
    data2_train, data2_test, label2_train, label2_test = train_test_split(
        data_2_all, label_2_all, test_size=test_number)
    data3_train, data3_test, label3_train, label3_test = train_test_split(
        data_3_all, label_3_all, test_size=test_number)

    return data1_train, data1_test, label1_train, label1_test, data2_train, data2_test, label2_train, label2_test, data3_train, data3_test, label3_train, label3_test


def svc_result(data_train, data_test, label_train, label_test):
    if data_train.shape[-1] == 0:
        return 0
    else:
        model = SVC(kernel='linear')
        model.fit(data_train, label_train)
        preditcion = model.predict(data_test)
        fpr, tpr, _ = roc_curve(
            label_test.ravel(), preditcion.ravel(), pos_label=1)
        roc_auc = auc(fpr, tpr)
        auprc = average_precision_score(label_test.ravel(), preditcion.ravel())
        return roc_auc, auprc


def whole_process(data_1_all, data_2_all, data_3_all, label_1_all, label_2_all,
                  label_3_all,n):
    roc1 = []
    auprc1 = []
    roc2 = []
    auprc2 = []
    roc3 = []
    auprc3 = []
    for itern in range(100):
        print(itern)
        chose1_p=[]
        chose2_p=[]
        chose3_p=[]
        data1_train, data1_test, label1_train, label1_test, data2_train, data2_test, label2_train, label2_test, data3_train, data3_test, label3_train, label3_test = cross_validation(
            data_1_all, data_2_all, data_3_all, label_1_all, label_2_all, label_3_all, 20)
        for i in range(data1_train.shape[1]):         
            if stats.ttest_ind(data1_train[:,i][label1_train==1],data1_train[:,i][label1_train==-1])[1]<0.05:
                chose1_p.append(i)
            if stats.ttest_ind(data2_train[:,i][label2_train==1],data2_train[:,i][label2_train==-1])[1]<0.05:
                chose2_p.append(i)
            if stats.ttest_ind(data3_train[:,i][label3_train==1],data3_train[:,i][label3_train==-1])[1]<0.05:
                chose3_p.append(i)
        temp1,temp2=svc_result(data1_train[:,chose1_p], data1_test[:,chose1_p], label1_train, label1_test)
        roc1.append(temp1)
        auprc1.append(temp2)
        temp1,temp2=svc_result(data2_train[:,chose2_p], data2_test[:,chose2_p], label2_train, label2_test)
        roc2.append(temp1)
        auprc2.append(temp2)
        temp1,temp2=svc_result(data3_train[:,chose3_p], data3_test[:,chose3_p], label3_train, label3_test)
        roc3.append(temp1)
        auprc3.append(temp2)       
    with open('bench2_f1.txt','w') as f:
        print('AUC','AUPRC',file=f)
        for i,j in zip(roc1,auprc1):
            print(i,j,file=f)
    with open('bench2_f2.txt','w') as f:
        print('AUC','AUPRC',file=f)
        for i,j in zip(roc2,auprc2):
            print(i,j,file=f)
    with open('bench2_f3.txt','w') as f:
        print('AUC','AUPRC',file=f)
        for i,j in zip(roc3,auprc3):
            print(i,j,file=f)   


if __name__ == "__main__":
    ### This is the parameter area
    n=100
    names = [
        'sample_BRCA.pkl', 'sample_BRCA_label.pkl', 'sample_OV.pkl', 'sample_OV_label.pkl', 'sample_PRAD.pkl',
        'sample_PRAD_label.pkl', 'sample_gene_names.pkl'
    ]
    data_1_all, data_2_all, data_3_all, label_1_all, label_2_all, label_3_all = read_data(
        names)
    whole_process(data_1_all, data_2_all, data_3_all, label_1_all, label_2_all,
                  label_3_all,n)
