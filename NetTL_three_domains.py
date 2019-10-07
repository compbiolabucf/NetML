#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, average_precision_score
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
np.random.seed(20)
np.set_printoptions(threshold=np.inf)


### This function reads data sets from pkl files directly, after that, data cleaning process will be excuated too
def read_data(names):
    data_1 = np.array(pickle.load(open(names[0], 'rb'))).T
    label_1 = np.array(pickle.load(open(names[1], 'rb')).flatten())
    data_2 = np.array(pickle.load(open(names[2], 'rb'))).T
    label_2 = np.array(pickle.load(open(names[3], 'rb')).flatten())
    data_3 = np.array(pickle.load(open(names[4], 'rb'))).T
    label_3 = np.array(pickle.load(open(names[5], 'rb')).flatten())
    #ppi = pickle.load(open(names[6], 'rb'))
    gene_names = pickle.load(open(names[7], 'rb'))
    '''
    temp_name_list = []
    for i in ppi.keys():
        temp_name_list.extend(ppi[i])
    ppi_names = set(ppi.keys()).union(set(temp_name_list))
    '''
    n = data_1.shape[1]

    percent = 0.5
    zero_col = []
    var_mean1 = []
    var_mean2 = []
    var_mean3 = []
    for i in range(n):
        if data_1.shape[0] - np.count_nonzero(data_1[:, i]) > 10:
            zero_col.append(i)
        if data_2.shape[0] - np.count_nonzero(data_2[:, i]) > 10:
            zero_col.append(i)
        if data_3.shape[0] - np.count_nonzero(data_3[:, i]) > 10:
            zero_col.append(i)
    zero_col = list(set(zero_col))
    data_1 = np.delete(data_1, zero_col, 1)
    data_2 = np.delete(data_2, zero_col, 1)
    data_3 = np.delete(data_3, zero_col, 1)
    print(data_1.shape)
    for i in sorted(zero_col, reverse=True):
        del gene_names[i]
    n1 = data_1.shape[1]
    for i in range(n1):
        var_mean1.append(np.var(data_1[:, i]) / np.mean(data_1[:, i]))
        var_mean2.append(np.var(data_2[:, i]) / np.mean(data_2[:, i]))
        var_mean3.append(np.var(data_3[:, i]) / np.mean(data_3[:, i]))
    high_rank1 = set(np.array(var_mean1).argsort()[-int(percent * n):][::-1])
    high_rank2 = set(np.array(var_mean2).argsort()[-int(percent * n):][::-1])
    high_rank3 = set(np.array(var_mean3).argsort()[-int(percent * n):][::-1])

    rest_pos = list(
        high_rank1.intersection(high_rank2.intersection(high_rank3)))
    data_1 = np.delete(data_1, rest_pos, 1)
    data_2 = np.delete(data_2, rest_pos, 1)
    data_3 = np.delete(data_3, rest_pos, 1)
    for i in sorted(rest_pos, reverse=True):
        del gene_names[i]
    '''
    final_gene_names = list(set(gene_names).intersection(ppi_names))

    for i in range(len(gene_names) - 1, -1, -1):
        if gene_names[i] not in final_gene_names:
            data_1 = np.delete(data_1, i, 1)
            data_2 = np.delete(data_2, i, 1)
            data_3 = np.delete(data_3, i, 1)
            del gene_names[i]
    ppi_matrix = np.zeros((len(gene_names), len(gene_names)))
    for i, j in enumerate(gene_names):
        if j in ppi.keys():
            for k in ppi[j]:
                if k in gene_names:
                    ppi_matrix[i, gene_names.index(k)] = 1 
    n = len(gene_names)
    for i in range(n):
        for j in range(n):
            if i != j:
                ppi_matrix[i, j] = np.maximum(ppi_matrix[i, j],
                                              ppi_matrix[j, i])
    empty_ppi = []
    for i in range(ppi_matrix.shape[0]):
        if np.sum(ppi_matrix[i, :]) == 0:
            empty_ppi.append(i)
    ppi_matrix = np.array(ppi_matrix)
    data_1 = np.delete(data_1, empty_ppi, 1)
    data_2 = np.delete(data_2, empty_ppi, 1)
    data_3 = np.delete(data_3, empty_ppi, 1)
    for i in sorted(empty_ppi, reverse=True):
        del gene_names[i]
    ppi_matrix = np.delete(ppi_matrix, empty_ppi, 1)
    ppi_matrix = np.delete(ppi_matrix, empty_ppi, 0)
    '''
    n = len(gene_names)
    ppi_matrix = np.ones((len(gene_names), len(gene_names)))
    return n, data_1, data_2, data_3, label_1, label_2, label_3, ppi_matrix, gene_names


### output the data matrices, n is the number of features


### The defination of the cross valation nethod
def cross_validation(data_1_all, data_2_all, label_1_all, label_2_all,
                     validation_number, test_number):
    ### get the test part
    data_1, data1_test, label_1, label1_test = train_test_split(
        data_1_all, label_1_all, test_size=test_number)
    data_2, data2_test, label_2, label2_test = train_test_split(
        data_2_all, label_2_all, test_size=test_number)
    data_3, data3_test, label_3, label3_test = train_test_split(
        data_3_all, label_3_all, test_size=test_number)
    ### get the train and validation parts
    data1_train, data1_validation, label1_train, label1_validation = train_test_split(
        data_1, label_1, test_size=validation_number)
    data2_train, data2_validation, label2_train, label2_validation = train_test_split(
        data_2, label_2, test_size=validation_number)
    data3_train, data3_validation, label3_train, label3_validation = train_test_split(
        data_3, label_3, test_size=validation_number)
    return data1_train, data1_validation, data1_test, label1_train,\
           label1_validation, label1_test, data2_train, data2_validation,\
           data2_test, label2_train, label2_validation, label2_test,\
           data3_train, data3_validation, data3_test, label3_train,\
           label3_validation, label3_test


### SVC area, which is used for the model1's classfication part
def svc_result(data_train, data_test, label_train, label_test):
    if data_train.shape[-1] == 0:
        return 0
    else:
        model = SVC()
        model.fit(data_train, label_train)
        preditcion = model.predict(data_test)
        fpr, tpr, _ = roc_curve(
            label_test.ravel(), preditcion.ravel(), pos_label=1)
        roc_auc = auc(fpr, tpr)
        auprc = average_precision_score(label_test.ravel(), preditcion.ravel())
        return roc_auc, auprc


### model1 area


## This is defination of laplace matrix
def laplace_matrix(n, data, ppi_matrix):
    cc_matrix = np.corrcoef(data.T)
    cc_matrix = np.abs(cc_matrix)
    cc_matrix = cc_matrix * ppi_matrix
    adj_matrix = cc_matrix - np.eye(n)
    adj_matrix[np.isnan(adj_matrix)] = 0.01
    normlize_matrix = np.diag(
        np.dot(adj_matrix,
               np.repeat(1., n).reshape((n, 1)).ravel()))
    
    normlize_matrix = np.linalg.inv(np.sqrt(normlize_matrix))
    laplace = np.eye(n) - np.dot(
        np.dot(normlize_matrix, adj_matrix), normlize_matrix)
    return laplace


## This is the defination of data y, which represents the relationship of the label and gene expression of each sample
def initial_y(n, data, label):
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.corrcoef(data.T[i, :], label.ravel())[1][0]
    y = np.abs(y)
    y[np.isnan(y)] = 0
    y = y.reshape(n, 1)
    return y


## this is the initialzation of feature selection list, all elements are set to be small amount
def initial_f(n):
    f1 = np.repeat(1. / n, n).reshape(n, 1)
    f2 = f1.copy()
    f3 = f1.copy()
    fc = f1.copy()
    return f1, f2, f3, fc


## In the approach process, this function is used to update the feature selection list fi. i=1,2
def model1_update_fi(n, laplace, fi, fc, y, alpha, gamma):
    Q = np.linalg.cholesky(alpha * laplace + (1 - alpha) * np.eye(n))
    Y = -(np.dot(fc.T, Q.T - (1 - alpha) * np.dot(y.T, np.linalg.inv(Q)))).T
    model = linear_model.Lasso(1 / 2 * gamma)

    model.fit(Q, Y)
    fi = model.coef_
    fi = fi.reshape((n, 1))
    return fi


## In the approach process, this function is used to update the common feature selection list fc.
def model1_update_fc(n, laplace1, laplace2, laplace3, f1, y1, f2, y2, f3, y3,
                     fc, alpha, gamma):
    Q = np.linalg.cholesky(alpha * laplace1 + 2 * (1 - alpha) * np.eye(n) +
                           alpha * laplace2)
    Y = -(np.dot((alpha * np.dot(f1.T, laplace1) + (1 - alpha) * (f1 - y1).T) +
                 (alpha * np.dot(f2.T, laplace2) + (1 - alpha) * (f2 - y2).T) +
                 (alpha * np.dot(f3.T, laplace3) +
                  (1 - alpha) * (f3 - y3).T), np.linalg.inv(Q))).T
    model = linear_model.Lasso(1 / 2 * gamma)
    model.fit(Q, Y)
    fc = model.coef_
    fc = fc.reshape((1, len(fc)))
    return fc


## Once the model 1 function generate the feature selection list already, this function can generate the roc value
## based on the list to evaluate the model's performance
def model1_roc(n, f1, f2, f3, data1, data2, data3, label1_train, label1_test,
               label2_train, label2_test, label3_train, label3_test):
    f1[f1 > 0] = 1
    f1[f1 <= 0] = 0
    f2[f2 > 0] = 1
    f2[f2 <= 0] = 0
    f3[f3 > 0] = 1
    f3[f3 <= 0] = 0
    if sum(f1)==0 or sum(f2)==0 or sum(f3) == 0:
        return 0, 0, 0, 0, 0, 0
    left1_sel = np.diag(f1.ravel())
    left2_sel = np.diag(f2.ravel())
    left3_sel = np.diag(f3.ravel())
    data1 = np.dot(left1_sel, data1.T)
    data2 = np.dot(left2_sel, data2.T)
    data3 = np.dot(left3_sel, data3.T)
    ind = []
    for i in range(n):
        temp1 = data1[i, :]
        if len(temp1[temp1 == 0]) > 20:
            ind.append(i)
    data1 = np.delete(data1, ind, axis=0)
    ind = []
    for i in range(n):
        temp2 = data2[i, :]
        if len(temp2[temp2 == 0]) > 20:
            ind.append(i)
    ind = []
    for i in range(n):
        temp3 = data3[i, :]
        if len(temp3[temp3 == 0]) > 20:
            ind.append(i)
    data3 = np.delete(data3, ind, axis=0)
    data1 = data1.T
    data1_train = data1[:len(label1_train), :]
    data1_test = data1[len(label1_train):, :]
    data2 = data2.T
    data2_train = data2[:len(label2_train), :]
    data2_test = data2[len(label2_train):, :]
    data3 = data3.T
    data3_train = data3[:len(label3_train), :]
    data3_test = data3[len(label3_train):, :]
    roc1, auprc1 = svc_result(data1_train, data1_test, label1_train,
                              label1_test)
    roc2, auprc2 = svc_result(data2_train, data2_test, label2_train,
                              label2_test)
    roc3, auprc3 = svc_result(data3_train, data3_test, label3_train,
                              label3_test)
    return roc1, roc2, roc3, auprc1, auprc2, auprc3


## this function trains model1 on the training data set, and output the roc values which is based on the model's
## performance on the validation data
def train_model1(n, data1_train, data1_validation, label1_train,
                 label1_validation, data2_train, data2_validation,
                 label2_train, label2_validation, data3_train,
                 data3_validation, label3_train, label3_validation, alpha,
                 gamma1, gamma2, gamma3, gamma4,gene_names):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    data3 = np.concatenate((data3_train, data3_validation), axis=0)
    laplace1 = laplace_matrix(n, data1, ppi_matrix)
    laplace2 = laplace_matrix(n, data2, ppi_matrix)
    laplace3 = laplace_matrix(n, data3, ppi_matrix)
    y1 = initial_y(n, data1_train, label1_train)
    y2 = initial_y(n, data2_train, label2_train)
    y3 = initial_y(n, data3_train, label3_train)
    f1, f2, f3, fc = initial_f(n)
    for interation in range(50):
        f1 = f1.reshape((n, 1))
        f2 = f2.reshape((n, 1))
        f3 = f3.reshape((n, 1))
        fc = fc.reshape((n, 1))
        f1_ori = f1.copy()
        f1 = model1_update_fi(n, laplace1, f1, fc, y1, alpha, gamma1)
        f2_ori = f2.copy()
        f2 = model1_update_fi(n, laplace2, f2, fc, y2, alpha, gamma2)
        f3_ori = f3.copy()
        f3 = model1_update_fi(n, laplace3, f3, fc, y3, alpha, gamma3)
        fc_ori = fc.copy()
        fc = model1_update_fc(n, laplace1, laplace2, laplace3, f1, y1, f2, y2,
                              f3, y3, fc, alpha, gamma4)
        if np.sum(np.abs(f1 - f1_ori)) > 100:
            break
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(f3 - f3_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    f1_basic=f1.copy()
    f2_basic=f2.copy()
    f3_basic=f3.copy()
    fc_basic=fc.copy()    
    roc1_train, roc2_train, roc3_train, auprc1_train, auprc2_train, auprc3_train = model1_roc(
        n, f1, f2, f3, data1, data2, data3, label1_train, label1_validation,
        label2_train, label2_validation, label3_train, label3_validation)
    if not os.path.exists('f1_select.txt'):
        if roc1_train>0.7 and roc2_train>0.7 and roc3_train>0.7 and np.sum(fc)>3:
            with open('f1_select.txt','w') as f:
                for loc, name in zip(f1_basic,gene_names):
                    if loc>0:
                        print(name, loc,file=f)
            with open('f2_select.txt','w') as f:
                for loc, name in zip(f2_basic,gene_names):
                    if loc>0:
                        print(name, loc,file=f)
            with open('f3_select.txt','w') as f:
                for loc, name in zip(f3_basic,gene_names):
                    if loc>0:
                        print(name, loc,file=f)
            with open('fc_select.txt','w') as f:
                for loc, name in zip(fc_basic,gene_names):
                    if loc>0:
                        print(name, loc,file=f)                   
                    
                    
    return roc1_train, roc2_train, roc3_train, auprc1_train, auprc2_train, auprc3_train


## this function trains the model1 on the same training data set and output its roc value
## to evaluate its performance on the test data sets
def test_model1(n, data1_train, data1_validation, data1_test, label1_train,
                label1_validation, label1_test, data2_train, data2_validation,
                data2_test, label2_train, label2_validation, label2_test,
                data3_train, data3_validation, data3_test, label3_train,
                label3_validation, label3_test, alpha, gamma1, gamma2, gamma3,
                gamma4):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    data3 = np.concatenate((data3_train, data3_validation), axis=0)
    data1_train = np.concatenate((data1, data1_test), axis=0)
    data2_train = np.concatenate((data2, data2_test), axis=0)
    data3_train = np.concatenate((data3, data3_test), axis=0)
    label1_train = np.concatenate((label1_train, label1_validation), axis=0)
    label2_train = np.concatenate((label2_train, label2_validation), axis=0)
    label3_train = np.concatenate((label3_train, label3_validation), axis=0)
    label1_temp = np.concatenate((label1_train.reshape(
        (len(label1_train), 1)), np.repeat(0., len(label1_test)).reshape(
            (len(label1_test), 1))),
                                 axis=0)
    label2_temp = np.concatenate((label2_train.reshape(len(label2_train), 1),
                                  np.repeat(0., len(label2_test)).reshape(
                                      (len(label2_test), 1))),
                                 axis=0)
    label3_temp = np.concatenate((label3_train.reshape(len(label3_train), 1),
                                  np.repeat(0., len(label3_test)).reshape(
                                      (len(label3_test), 1))),
                                 axis=0)
    laplace1 = laplace_matrix(n, data1, ppi_matrix)
    laplace2 = laplace_matrix(n, data2, ppi_matrix)
    laplace3 = laplace_matrix(n, data3, ppi_matrix)
    y1 = initial_y(n, data1_train, label1_temp)
    y2 = initial_y(n, data2_train, label2_temp)
    y3 = initial_y(n, data3_train, label3_temp)
    f1, f2, f3, fc = initial_f(n)
    for interation in range(50):
        f1 = f1.reshape((n, 1))
        f2 = f2.reshape((n, 1))
        f3 = f3.reshape((n, 1))
        fc = fc.reshape((n, 1))
        f1_ori = f1.copy()
        f1 = model1_update_fi(n, laplace1, f1, fc, y1, alpha, gamma1)
        f2_ori = f2.copy()
        f2 = model1_update_fi(n, laplace2, f2, fc, y2, alpha, gamma2)
        f3_ori = f3.copy()
        f3 = model1_update_fi(n, laplace3, f3, fc, y3, alpha, gamma3)
        fc_ori = fc.copy()
        fc = model1_update_fc(n, laplace1, laplace2, laplace3, f1, y1, f2, y2,
                              f3, y3, fc, alpha, gamma4)
        if np.sum(np.abs(f1 - f1_ori)) > 100:
            break
        if np.sum(np.abs(f1 - f1_ori)) < 1e-5 or \
        np.sum(np.abs(f2 - f2_ori)) < 1e-5 or \
        np.sum(np.abs(f3 - f3_ori)) < 1e-5 or \
        np.sum(np.abs(fc - fc_ori)) < 1e-5:
            break
    roc1_test, roc2_test, roc3_test, auprc1_test, auprc2_test, auprc3_test = model1_roc(
        n, f1, f2, f3, data1_train, data2_train, data3_train, label1_train,
        label1_test, label2_train, label2_test, label3_train, label3_test)
    return roc1_test, roc2_test, roc3_test, auprc1_test, auprc2_test, auprc3_test


### model1 main area
## this function utilize the functions above to output all the roc values of model1 based on the training data sets,
## validation data sets, test data sets


def model1(n, data1_train, data1_validation, data1_test, label1_train,
           label1_validation, label1_test, data2_train, data2_validation,
           data2_test, label2_train, label2_validation, label2_test,
           data3_train, data3_validation, data3_test, label3_train,
           label3_validation, label3_test, alpha, gamma1, gamma2, gamma3,
           gamma4,gene_names):
    roc1_train, roc2_train, roc3_train, auprc1_train, auprc2_train, auprc3_train = train_model1(
        n, data1_train, data1_validation, label1_train, label1_validation,
        data2_train, data2_validation, label2_train, label2_validation,
        data3_train, data3_validation, label3_train, label3_validation, alpha,
        gamma1, gamma2, gamma3, gamma4,gene_names)
    roc1_test, roc2_test, roc3_test, auprc1_test, auprc2_test, auprc3_test = test_model1(
        n, data1_train, data1_validation, data1_test, label1_train,
        label1_validation, label1_test, data2_train, data2_validation,
        data2_test, label2_train, label2_validation, label2_test, data3_train,
        data3_validation, data3_test, label3_train, label3_validation,
        label3_test, alpha, gamma1, gamma2, gamma3, gamma4)
    return roc1_train, roc2_train, roc3_train, auprc1_train, auprc2_train, auprc3_train, roc1_test, roc2_test, roc3_test, auprc1_test, auprc2_test, auprc3_test


####################
## This function runs model1 all over the possible parameters which are defined
def model1_roc_all(n, data1_train, data1_validation, data1_test, label1_train,
                   label1_validation, label1_test, data2_train,
                   data2_validation, data2_test, label2_train,
                   label2_validation, label2_test, data3_train,
                   data3_validation, data3_test, label3_train,
                   label3_validation, label3_test,gene_names):
    alpha = 0.1
    gamma1_all = np.arange(5e-7, 16e-7, 5e-7)
    gamma2_all = np.arange(5e-7, 16e-7, 5e-7)
    gamma3_all = np.arange(5e-7, 16e-7, 5e-7)
    gamma4_all = np.arange(5e-7, 16e-7, 5e-7)
    model1_roc_train = np.zeros((3, 3, 3, 3))
    model2_roc_train = np.zeros((3, 3, 3, 3))
    model3_roc_train = np.zeros((3, 3, 3, 3))
    model1_roc_test = np.zeros((3, 3, 3, 3))
    model2_roc_test = np.zeros((3, 3, 3, 3))
    model3_roc_test = np.zeros((3, 3, 3, 3))
    model1_auprc_test = np.zeros((3, 3, 3, 3))
    model2_auprc_test = np.zeros((3, 3, 3, 3))
    model3_auprc_test = np.zeros((3, 3, 3, 3))
    for iii in range(3):
        for jjj in range(3):
            for kkk in range(3):
                for mmm in range(3):
                    roc1_train, roc2_train, roc3_train, auprc1_train, auprc2_train, auprc3_train, roc1_test, roc2_test, roc3_test, auprc1_test, auprc2_test, auprc3_test = model1(
                        n, data1_train, data1_validation, data1_test,
                        label1_train, label1_validation, label1_test,
                        data2_train, data2_validation, data2_test,
                        label2_train, label2_validation, label2_test,
                        data3_train, data3_validation, data3_test,
                        label3_train, label3_validation, label3_test, alpha,
                        gamma1_all[iii], gamma2_all[jjj], gamma3_all[kkk],
                        gamma4_all[mmm],gene_names)
                    model1_roc_train[iii, jjj, kkk, mmm] += roc1_train
                    model2_roc_train[iii, jjj, kkk, mmm] += roc2_train
                    model3_roc_train[iii, jjj, kkk, mmm] += roc3_train
                    model1_roc_test[iii, jjj, kkk, mmm] += roc1_test
                    model2_roc_test[iii, jjj, kkk, mmm] += roc2_test
                    model3_roc_test[iii, jjj, kkk, mmm] += roc3_test
                    model1_auprc_test[iii, jjj, kkk, mmm] += auprc1_test
                    model2_auprc_test[iii, jjj, kkk, mmm] += auprc2_test
                    model3_auprc_test[iii, jjj, kkk, mmm] += auprc3_test
    return model1_roc_train, model2_roc_train, model3_roc_train, model1_roc_test, model2_roc_test, model3_roc_test, model1_auprc_test, model2_auprc_test, model3_auprc_test


### This function records all models' results
def all_models(n, data1_train, data1_validation, data1_test, label1_train,
               label1_validation, label1_test, data2_train, data2_validation,
               data2_test, label2_train, label2_validation, label2_test,
               data3_train, data3_validation, data3_test, label3_train,
               label3_validation, label3_test,gene_names):

    # model1 part
    model1_roc_train, model2_roc_train, model3_roc_train, model1_roc_test, model2_roc_test, model3_roc_test, model1_auprc_test, model2_auprc_test, model3_auprc_test = model1_roc_all(
        n, data1_train, data1_validation, data1_test, label1_train,
        label1_validation, label1_test, data2_train, data2_validation,
        data2_test, label2_train, label2_validation, label2_test, data3_train,
        data3_validation, data3_test, label3_train, label3_validation,
        label3_test,gene_names)
    loc1 = np.argmax(model1_roc_train)
    model1_roc_train_max = np.max(model1_roc_train)
    final_model1_roc = model1_roc_test[loc1 // 27][loc1 % 27 // 9][
        loc1 % 27 % 9 // 3][loc1 % 27 % 9 % 3]
    final_model1_auprc = model1_auprc_test[loc1 // 27][loc1 % 27 // 9][
        loc1 % 27 % 9 // 3][loc1 % 27 % 9 % 3]

    loc2 = np.argmax(model2_roc_train)
    model2_roc_train_max = np.max(model2_roc_train)
    final_model2_roc = model2_roc_test[loc2 // 27][loc2 % 27 // 9][
        loc2 % 27 % 9 // 3][loc2 % 27 % 9 % 3]
    final_model2_auprc = model2_auprc_test[loc2 // 27][loc2 % 27 // 9][
        loc2 % 27 % 9 // 3][loc2 % 27 % 9 % 3]

    loc3 = np.argmax(model3_roc_train)
    model3_roc_train_max = np.max(model3_roc_train)
    final_model3_roc = model3_roc_test[loc3 // 27][loc3 % 27 // 9][
        loc3 % 27 % 9 // 3][loc3 % 27 % 9 % 3]
    final_model3_auprc = model3_auprc_test[loc3 // 27][loc3 % 27 // 9][
        loc3 % 27 % 9 // 3][loc3 % 27 % 9 % 3]

    return model1_roc_train_max, final_model1_roc, final_model1_auprc, model2_roc_train_max, final_model2_roc, final_model2_auprc, model3_roc_train_max, final_model3_roc, final_model3_auprc


### This function is the cross validation process, it runs the CV process 50 times, and output every time's result
### in the txt file
def whole_process(n, data_1_all, data_2_all, data_3_all, label_1_all,
                  label_2_all, label_3_all, ppi, validation_number,
                  test_number,gene_names):
    ### run 50 times, get the average
    model1_roc_all = 0.
    model2_roc_all = 0.
    model3_roc_all = 0.

    model1_auprc_all = 0.
    model2_auprc_all = 0.
    model3_auprc_all = 0.

    model1_roc_train_max_all = 0.
    model2_roc_train_max_all = 0.
    model3_roc_train_max_all = 0.

    for i in range(20):
        print(i)
        data1_train, data1_validation, data1_test, label1_train, label1_validation,\
                  label1_test, data2_train, data2_validation, data2_test, label2_train,\
                  label2_validation, label2_test, data3_train, data3_validation, data3_test,\
                  label3_train, label3_validation, label3_test=cross_validation(data_1_all,data_2_all,\
                  label_1_all,label_2_all,validation_number,test_number)

        model1_roc_train_max, final_model1_roc, final_model1_auprc, model2_roc_train_max, final_model2_roc, final_model2_auprc, model3_roc_train_max, final_model3_roc, final_model3_auprc = all_models(
            n, data1_train, data1_validation, data1_test, label1_train,
            label1_validation, label1_test, data2_train, data2_validation,
            data2_test, label2_train, label2_validation, label2_test,
            data3_train, data3_validation, data3_test, label3_train,
            label3_validation, label3_test,gene_names)

        model1_roc_train_max_all += model1_roc_train_max
        model1_roc_all += final_model1_roc
        model1_auprc_all += final_model1_auprc

        model2_roc_train_max_all += model2_roc_train_max
        model2_roc_all += final_model2_roc
        model2_auprc_all += final_model2_auprc

        model3_roc_train_max_all += model3_roc_train_max
        model3_roc_all += final_model3_roc
        model3_auprc_all += final_model3_auprc
        print(i)
        temp_report = open('temp_report_model1_7.txt', 'a')
        print(file=temp_report)
        print('iterations:', i + 1, file=temp_report)
        print('for f1', file=temp_report)
        print(model1_roc_train_max, file=temp_report)
        print(final_model1_roc, file=temp_report)
        print(final_model1_auprc, file=temp_report)
        print('for f2', file=temp_report)
        print(model2_roc_train_max, file=temp_report)
        print(final_model2_roc, file=temp_report)
        print(final_model2_auprc, file=temp_report)
        print('for f3', file=temp_report)
        print(model3_roc_train_max, file=temp_report)
        print(final_model3_roc, file=temp_report)
        print(final_model3_auprc, file=temp_report)
        temp_report.close()
    final_report = open('final_report_model1_7.txt', 'w')
    print('for f1', file=final_report)
    print(model1_roc_train_max_all / (i + 1), file=final_report)
    print(model1_roc_all / (i + 1), file=final_report)
    print(model1_auprc_all / (i + 1), file=final_report)
    print('for f2', file=final_report)
    print(model2_roc_train_max_all / (i + 1), file=final_report)
    print(model2_roc_all / (i + 1), file=final_report)
    print(model2_auprc_all / (i + 1), file=final_report)
    print('for f3', file=final_report)
    print(model3_roc_train_max_all / (i + 1), file=final_report)
    print(model3_roc_all / (i + 1), file=final_report)
    print(model3_auprc_all / (i + 1), file=final_report)
    final_report.close()


###  main area

if __name__ == "__main__":
    ### This is the parameter area
    '''
    names=[]
    for i in range(1,len(sys.argv)):
        names.append(sys.argv[i])
    '''

    names = [
        'sample_BRCA.pkl', 'sample_BRCA_label.pkl', 'sample_OV.pkl', 'sample_OV_label.pkl', 'sample_PRAD.pkl',
        'sample_PRAD_label.pkl', 'sample_ppi.pkl', 'sample_gene_names.pkl'
    ]

    test_number = 10
    validation_number = 10
    n, data_1_all, data_2_all, data_3_all, label_1_all, label_2_all, label_3_all, ppi_matrix, gene_names = read_data(
        names)

    whole_process(n, data_1_all, data_2_all, data_3_all, label_1_all,
                  label_2_all, label_3_all, ppi_matrix, validation_number,
                  test_number,gene_names)
