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
    gene_names = pickle.load(open(names[6], 'rb'))
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

    n = len(gene_names)
    return n, data_1, data_2, data_3, label_1, label_2, label_3, gene_names


### output the data matrices, n is the number of features


### The defination of the cross valation nethod
def cross_validation(data_1_all, data_2_all, data_3_all, label_1_all,
                     label_2_all, label_3_all, validation_number, test_number):
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


## This is the defination of data y, which represents the relationship of the label and gene expression of each sample
def initial_y(n, data, label):
    y = np.zeros(n)
    for i in range(n):
        y[i] = np.corrcoef(data.T[i, :], label.ravel())[1][0]
    y = np.abs(y)
    y[np.isnan(y)] = 0
    y = y.reshape(n, 1)
    return y


### model2 area
## this function defines the adjance matrix of model2 which is different from the model1's
def adjance_matrix(data):
    v = data.shape[0]
    u = data.shape[1]
    left_matrix = np.eye(v, v)
    right_matrix = np.eye(u, u)
    left_matrix = np.dot(data, np.repeat(1., u).reshape((u, 1)).ravel())
    right_matrix = np.dot(np.repeat(1., v).reshape((1, v)).ravel(), data)
    left_matrix[left_matrix == 0] = 0.001
    right_matrix[right_matrix == 0] = 0.001
    left_matrix = np.diag(left_matrix)
    right_matrix = np.diag(right_matrix)
    left_matrix = np.linalg.inv(np.sqrt(left_matrix))
    right_matrix = np.linalg.inv(np.sqrt(right_matrix))
    adjance = np.dot(np.dot(left_matrix, data), right_matrix)
    return adjance.T


## This function initialized all the feature selection lists and lables
def initial_fv_fu(data1, data2, data3, label1_train, label2_train,
                  label3_train):
    n = data1.shape[1]
    m1 = data1.shape[0]
    m2 = data2.shape[0]
    m3 = data3.shape[0]
    fv1 = np.repeat(1. / n, n).reshape((n, 1))
    fv2 = fv1.copy()
    fv3 = fv1.copy()
    fvc = fv1.copy()
    fu1 = np.concatenate((label1_train.reshape(len(label1_train), 1),
                          np.repeat(0., m1 - label1_train.shape[0]).reshape(
                              (m1 - label1_train.shape[0], 1))),
                         axis=0)
    fu2 = np.concatenate((label2_train.reshape(len(label2_train), 1),
                          np.repeat(0., m2 - label2_train.shape[0]).reshape(
                              (m2 - label2_train.shape[0], 1))),
                         axis=0)
    fu3 = np.concatenate((label3_train.reshape(len(label3_train), 1),
                          np.repeat(0., m3 - label3_train.shape[0]).reshape(
                              (m3 - label3_train.shape[0], 1))),
                         axis=0)
    return fv1, fv2, fv3, fvc, fu1, fu2, fu3


## In the approach process, this function is used to update the feature selection list fv. v=1,2
def model2_update_fvi(n, adjance, fui, yvi, fvc, alpha, gamma):
    Y = (np.dot(adjance, fui) + alpha * yvi) / (1 + alpha) - fvc
    model = linear_model.Lasso(1 / 2 * gamma)
    Q = np.eye(n)
    model.fit(Q, Y)
    fvi = model.coef_
    fvi = fvi.reshape((n, 1))
    return fvi


## In the approach process, this function is used to update the common feature selection list fc.
def model2_update_fvc(n, adjance1, adjance2, adjance3, fu1, fu2, fu3, fv1, fv2,
                      fv3, yv1, yv2, yv3, alpha, gamma):
    Y = 1 / 2 * ((np.dot(adjance1, fu1) + alpha * yv1) / (1 + alpha) - fv1 +
                 (np.dot(adjance2, fu2) + alpha * yv2) / (1 + alpha) - fv2 +
                 (np.dot(adjance3, fu3) + alpha * yv3) / (1 + alpha) - fv3)
    model = linear_model.Lasso(1 / 2 * gamma)
    Q = np.eye(n)
    model.fit(Q, Y)
    fvc = model.coef_
    fvc = fvc.reshape((n, 1))
    return fvc


## In the approach process, this function is used to update the label's prediction.
def model2_update_fui(n, adjance, fvi, fvc, alpha, yui):
    fui = (np.dot(adjance.T, fvi) + np.dot(adjance.T, fvc) + alpha * yui) / (
        1 + alpha)
    return fui


## this function trains model2 with given data sets and defined parameters, and return
## the roc values of model on validation data sets
def train_model(n, data1_train, data1_validation, label1_train,
                label1_validation, data2_train, data2_validation, label2_train,
                label2_validation, data3_train, data3_validation, label3_train,
                label3_validation, alpha, gamma1, gamma2, gamma3, gamma4,gene_names):
    data1 = np.concatenate((data1_train, data1_validation), axis=0)
    data2 = np.concatenate((data2_train, data2_validation), axis=0)
    data3 = np.concatenate((data3_train, data3_validation), axis=0)
    fv1, fv2, fv3, fvc, fu1, fu2, fu3 = initial_fv_fu(
        data1, data2, data3, label1_train, label2_train, label3_train)
    label1_temp = np.concatenate(
        (label1_train.reshape(len(label1_train), 1),
         np.repeat(0., len(label1_validation)).reshape(
             (len(label1_validation)), 1)),
        axis=0)
    label2_temp = np.concatenate(
        (label2_train.reshape(len(label2_train), 1),
         np.repeat(0., len(label2_validation)).reshape(
             (len(label2_validation)), 1)),
        axis=0)
    label3_temp = np.concatenate(
        (label3_train.reshape(len(label3_train), 1),
         np.repeat(0., len(label3_validation)).reshape(
             (len(label3_validation)), 1)),
        axis=0)
    yv1 = initial_y(n, data1, label1_temp)
    yv2 = initial_y(n, data2, label2_temp)
    yv3 = initial_y(n, data3, label3_temp)
    adjance1 = adjance_matrix(data1)
    adjance2 = adjance_matrix(data2)
    adjance3 = adjance_matrix(data3)
    yu1 = fu1.copy()
    yu2 = fu2.copy()
    yu3 = fu3.copy()
    for i in range(50):
        fv1 = fv1.reshape((n, 1))
        fv2 = fv2.reshape((n, 1))
        fv3 = fv3.reshape((n, 1))
        fvc = fvc.reshape((n, 1))
        fv1_ori = fv1.copy()
        fv1 = model2_update_fvi(n, adjance1, fu1, yv1, fvc, alpha, gamma1)
        fv2_ori = fv2.copy()
        fv2 = model2_update_fvi(n, adjance2, fu2, yv2, fvc, alpha, gamma2)
        fv3_ori = fv3.copy()
        fv3 = model2_update_fvi(n, adjance3, fu3, yv3, fvc, alpha, gamma3)
        fvc_ori = fvc.copy()
        fvc = model2_update_fvc(n, adjance1, adjance2, adjance3, fu1, fu2, fu3,
                                fv1, fv2, fv3, yv1, yv2, yv3, alpha, gamma4)
        fu1 = model2_update_fui(n, adjance1, fv1, fvc, alpha, yu1)
        fu2 = model2_update_fui(n, adjance2, fv2, fvc, alpha, yu2)
        fu3 = model2_update_fui(n, adjance3, fv3, fvc, alpha, yu3)
        if np.sum(np.abs(fv1 - fv1_ori)) > 100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fv3 - fv3_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break
    fu1_test = fu1[len(fu1) - len(label1_validation):]
    fpr, tpr, _ = roc_curve(
        label1_validation.ravel(), fu1_test.ravel(), pos_label=1)
    roc_auc1 = auc(fpr, tpr)
    auprc1 = average_precision_score(label1_validation.ravel(),
                                     fu1_test.ravel())

    fu2_test = fu2[len(fu2) - len(label2_validation):]
    fpr, tpr, _ = roc_curve(
        label2_validation.ravel(), fu2_test.ravel(), pos_label=1)
    roc_auc2 = auc(fpr, tpr)
    auprc2 = average_precision_score(label2_validation.ravel(),
                                     fu2_test.ravel())

    fu3_test = fu3[len(fu3) - len(label3_validation):]
    fpr, tpr, _ = roc_curve(
        label3_validation.ravel(), fu3_test.ravel(), pos_label=1)
    roc_auc3 = auc(fpr, tpr)
    auprc3 = average_precision_score(label3_validation.ravel(),
                                     fu3_test.ravel())
    fv2 = fv2.reshape((n))
    fv2_val=fv2.copy()
    for i in range(len(fv2)):
        if fv2[i]>0:
            fv2[i]=1
        else:
            fv2[i]=0
    
    if roc_auc1>0. and roc_auc2>0.7 and roc_auc3>0. and sum(fv2)>200 and sum(fv2)<3000:
        print(roc_auc1,roc_auc2,roc_auc3)
        print('real:',roc_auc1,roc_auc2,roc_auc3)
        fv1 = fv1.reshape((n))
        fv2 = fv2.reshape((n))
        fv3 = fv3.reshape((n))
        fvc = fvc.reshape((n))
        '''
        with open('model2_f1_1.txt','a') as f:
            for i,j in zip(fv1,gene_names):
                print(i,j, file=f)
        with open('model2_f2_1.txt','a') as f:
            for i,j in zip(fv2_val,gene_names):
                print(i,j, file=f)
        with open('model2_f3_1.txt','a') as f:
            for i,j in zip(fv3,gene_names):
                print(i,j, file=f)
        with open('model2_fc_1.txt','a') as f:
            for i,j in zip(fvc,gene_names):
                print(i,j, file=f)
        '''
    return roc_auc1, roc_auc2, roc_auc2, auprc1, auprc2, auprc3


## This function gets the model2's performance on test data sets with give parameters
def test_model(n, data1_train, data1_validation, data1_test, label1_train,
               label1_validation, label1_test, data2_train, data2_validation,
               data2_test, label2_train, label2_validation, label2_test,
               data3_train, data3_validation, data3_test, label3_train,
               label3_validation, label3_test, alpha, gamma1, gamma2, gamma3,
               gamma4):
    data1 = np.concatenate((data1_train, data1_test), axis=0)
    data2 = np.concatenate((data2_train, data2_test), axis=0)
    data3 = np.concatenate((data3_train, data3_test), axis=0)
    label1_temp = np.concatenate((label1_train.reshape(len(label1_train), 1),
                                  np.repeat(0., len(label1_test)).reshape(
                                      (len(label1_test)), 1)),
                                 axis=0)
    label2_temp = np.concatenate((label2_train.reshape(len(label2_train), 1),
                                  np.repeat(0., len(label2_test)).reshape(
                                      (len(label2_test)), 1)),
                                 axis=0)
    label3_temp = np.concatenate((label3_train.reshape(len(label3_train), 1),
                                  np.repeat(0., len(label3_test)).reshape(
                                      (len(label3_test)), 1)),
                                 axis=0)
    fv1, fv2, fv3, fvc, fu1, fu2, fu3, = initial_fv_fu(
        data1, data2, data3, label1_train, label2_train, label3_train)
    yv1 = initial_y(n, data1, label1_temp)
    yv2 = initial_y(n, data2, label2_temp)
    yv3 = initial_y(n, data3, label3_temp)
    adjance1 = adjance_matrix(data1)
    adjance2 = adjance_matrix(data2)
    adjance3 = adjance_matrix(data3)
    yu1 = fu1.copy()
    yu2 = fu2.copy()
    yu3 = fu3.copy()
    for i in range(50):
        fv1 = fv1.reshape((n, 1))
        fv2 = fv2.reshape((n, 1))
        fv3 = fv3.reshape((n, 1))
        fvc = fvc.reshape((n, 1))
        fv1_ori = fv1.copy()
        fv1 = model2_update_fvi(n, adjance1, fu1, yv1, fvc, alpha, gamma1)
        fv2_ori = fv2.copy()
        fv2 = model2_update_fvi(n, adjance2, fu2, yv2, fvc, alpha, gamma2)
        fv3_ori = fv3.copy()
        fv3 = model2_update_fvi(n, adjance3, fu3, yv3, fvc, alpha, gamma3)
        fvc_ori = fvc.copy()
        fvc = model2_update_fvc(n, adjance1, adjance2, adjance3, fu1, fu2, fu3,
                                fv1, fv2, fv3, yv1, yv2, yv3, alpha, gamma4)
        fu1 = model2_update_fui(n, adjance1, fv1, fvc, alpha, yu1)
        fu2 = model2_update_fui(n, adjance2, fv2, fvc, alpha, yu2)
        fu3 = model2_update_fui(n, adjance3, fv3, fvc, alpha, yu3)
        if np.sum(np.abs(fv1 - fv1_ori)) > 100:
            break
        if np.sum(np.abs(fv1 - fv1_ori)) < 1e-5 or \
        np.sum(np.abs(fv2 - fv2_ori)) < 1e-5 or \
        np.sum(np.abs(fv3 - fv3_ori)) < 1e-5 or \
        np.sum(np.abs(fvc - fvc_ori)) < 1e-5:
            break

    fu1_test = fu1[len(fu1) - len(label1_test):]
    fpr, tpr, _ = roc_curve(label1_test.ravel(), fu1_test.ravel(), pos_label=1)
    roc_auc1 = auc(fpr, tpr)
    auprc1 = average_precision_score(label1_test.ravel(), fu1_test.ravel())

    fu2_test = fu2[len(fu2) - len(label2_test):]
    fpr, tpr, _ = roc_curve(label2_test.ravel(), fu2_test.ravel(), pos_label=1)
    roc_auc2 = auc(fpr, tpr)
    auprc2 = average_precision_score(label2_test.ravel(), fu2_test.ravel())

    fu3_test = fu3[len(fu3) - len(label3_test):]
    fpr, tpr, _ = roc_curve(label3_test.ravel(), fu3_test.ravel(), pos_label=1)
    roc_auc3 = auc(fpr, tpr)
    auprc3 = average_precision_score(label3_test.ravel(), fu3_test.ravel())

    return roc_auc1, roc_auc2, roc_auc3, auprc1, auprc2, auprc3


## this function recorders all the roc values of model2 on validation data and test data
def model2(n, data1_train, data1_validation, data1_test, label1_train,
           label1_validation, label1_test, data2_train, data2_validation,
           data2_test, label2_train, label2_validation, label2_test,
           data3_train, data3_validation, data3_test, label3_train,
           label3_validation, label3_test, alpha, gamma1, gamma2, gamma3,
           gamma4,gene_names):
    train_roc1, train_roc2, train_roc3, train_auprc1, train_auprc2, train_auprc3 = train_model(
        n, data1_train, data1_validation, label1_train, label1_validation,
        data2_train, data2_validation, label2_train, label2_validation,
        data3_train, data3_validation, label3_train, label3_validation, alpha,
        gamma1, gamma2, gamma3, gamma4,gene_names)
    # test_roc1, test_roc2, test_roc3, test_auprc1, test_auprc2, test_auprc3=0,0,0,0,0,0
    
    test_roc1, test_roc2, test_roc3, test_auprc1, test_auprc2, test_auprc3 = test_model(
        n, data1_train, data1_validation, data1_test, label1_train,
        label1_validation, label1_test, data2_train, data2_validation,
        data2_test, label2_train, label2_validation, label2_test, data3_train,
        data3_validation, data3_test, label3_train, label3_validation,
        label3_test, alpha, gamma1, gamma2, gamma3, gamma4)
    
    return train_roc1, train_roc2, train_roc3, train_auprc1, train_auprc2, train_auprc3, test_roc1, test_roc2, test_roc3, test_auprc1, test_auprc2, test_auprc3


## This function records best roc values with given all possible parameters
def model2_roc_all(n, data1_train, data1_validation, data1_test, label1_train,
                   label1_validation, label1_test, data2_train,
                   data2_validation, data2_test, label2_train,
                   label2_validation, label2_test, data3_train,
                   data3_validation, data3_test, label3_train,
                   label3_validation, label3_test,gene_names):
    alpha = 0.01
    gamma1_all = np.arange(5e-7, 16e-7, 5e-7)
    gamma2_all = np.arange(5e-8, 16e-8, 5e-8)
    gamma3_all = np.arange(5e-7, 16e-7, 5e-7)
    gamma4_all = np.arange(5e-8, 16e-8, 5e-8)
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
                    train_roc1, train_roc2, train_roc3, train_auprc1, train_auprc2, train_auprc3, test_roc1, test_roc2, test_roc3, test_auprc1, test_auprc2, test_auprc3 = model2(
                        n, data1_train, data1_validation, data1_test,
                        label1_train, label1_validation, label1_test,
                        data2_train, data2_validation, data2_test,
                        label2_train, label2_validation, label2_test,
                        data3_train, data3_validation, data3_test,
                        label3_train, label3_validation, label3_test, alpha,
                        gamma1_all[iii], gamma2_all[jjj], gamma3_all[kkk],
                        gamma4_all[mmm],gene_names)
                    model1_roc_train[iii, jjj, kkk, mmm] += train_roc1
                    model2_roc_train[iii, jjj, kkk, mmm] += train_roc2
                    model3_roc_train[iii, jjj, kkk, mmm] += train_roc3
                    model1_roc_test[iii, jjj, kkk, mmm] += test_roc1
                    model2_roc_test[iii, jjj, kkk, mmm] += test_roc2
                    model3_roc_test[iii, jjj, kkk, mmm] += test_roc3
                    model1_auprc_test[iii, jjj, kkk, mmm] += test_auprc1
                    model2_auprc_test[iii, jjj, kkk, mmm] += test_auprc2
                    model3_auprc_test[iii, jjj, kkk, mmm] += test_auprc3
    return model1_roc_train, model2_roc_train, model3_roc_train, model1_roc_test, model2_roc_test, model3_roc_test, model1_auprc_test, model2_auprc_test, model3_auprc_test


### This function records all models' results
def all_models(n, data1_train, data1_validation, data1_test, label1_train,
               label1_validation, label1_test, data2_train, data2_validation,
               data2_test, label2_train, label2_validation, label2_test,
               data3_train, data3_validation, data3_test, label3_train,
               label3_validation, label3_test,gene_names):

    # model2 part
    model1_roc_train, model2_roc_train, model3_roc_train, model1_roc_test, model2_roc_test, model3_roc_test, model1_auprc_test, model2_auprc_test, model3_auprc_test = model2_roc_all(
        n, data1_train, data1_validation, data1_test, label1_train,
        label1_validation, label1_test, data2_train, data2_validation,
        data2_test, label2_train, label2_validation, label2_test, data3_train,
        data3_validation, data3_test, label3_train, label3_validation,
        label3_test,gene_names)
    loc1 = np.argmin(model1_roc_train)
    model1_roc_train_max = np.max(model1_roc_train)
    final_model1_roc = model1_roc_test[loc1 // 27][loc1 % 27 // 9][
        loc1 % 27 % 9 // 3][loc1 % 27 % 9 % 3]
    final_model1_auprc = model1_auprc_test[loc1 // 27][loc1 % 27 // 9][
        loc1 % 27 % 9 // 3][loc1 % 27 % 9 % 3]

    loc2 = np.argmin(model2_roc_train)
    model2_roc_train_max = np.max(model2_roc_train)
    final_model2_roc = model2_roc_test[loc2 // 27][loc2 % 27 // 9][
        loc2 % 27 % 9 // 3][loc2 % 27 % 9 % 3]
    final_model2_auprc = model2_auprc_test[loc2 // 27][loc2 % 27 // 9][
        loc2 % 27 % 9 // 3][loc2 % 27 % 9 % 3]

    loc3 = np.argmin(model3_roc_train)
    model3_roc_train_max = np.max(model3_roc_train)
    final_model3_roc = model3_roc_test[loc3 // 27][loc3 % 27 // 9][
        loc3 % 27 % 9 // 3][loc3 % 27 % 9 % 3]
    final_model3_auprc = model3_auprc_test[loc3 // 27][loc3 % 27 // 9][
        loc3 % 27 % 9 // 3][loc3 % 27 % 9 % 3]

    return model1_roc_train_max, final_model1_roc, final_model1_auprc, model2_roc_train_max, final_model2_roc, final_model2_auprc, model3_roc_train_max, final_model3_roc, final_model3_auprc


### This function is the cross validation process, it runs the CV process 50 times, and output every time's result
### in the txt file
def whole_process(n, data_1_all, data_2_all, data_3_all, label_1_all,
                  label_2_all, label_3_all, validation_number, test_number,gene_names):
    ### run 50 times, get the average

    final_model1_roc_all = 0.
    final_model2_roc_all = 0.
    final_model3_roc_all = 0.

    model1_roc_train_max_all = 0.
    model2_roc_train_max_all = 0.
    model3_roc_train_max_all = 0.

    model1_auprc_max_all = 0.
    model2_auprc_max_all = 0.
    model3_auprc_max_all = 0.
    for i in range(20):
        print(i)
        data1_train, data1_validation, data1_test, label1_train, label1_validation, label1_test, data2_train, data2_validation, data2_test, label2_train, label2_validation, label2_test, data3_train, data3_validation, data3_test, label3_train, label3_validation, label3_test = cross_validation(
            data_1_all, data_2_all, data_3_all, label_1_all, label_2_all,
            label_3_all, validation_number, test_number)

        model1_roc_train_max, final_model1_roc, final_model1_auprc, model2_roc_train_max, final_model2_roc, final_model2_auprc, model3_roc_train_max, final_model3_roc, final_model3_auprc = all_models(
            n, data1_train, data1_validation, data1_test, label1_train,
            label1_validation, label1_test, data2_train, data2_validation,
            data2_test, label2_train, label2_validation, label2_test,
            data3_train, data3_validation, data3_test, label3_train,
            label3_validation, label3_test,gene_names)

        model1_roc_train_max_all += model1_roc_train_max
        final_model1_roc_all += final_model1_roc
        model1_auprc_max_all += final_model1_auprc

        model2_roc_train_max_all += model2_roc_train_max
        final_model2_roc_all += final_model2_roc
        model2_auprc_max_all += final_model2_auprc

        model3_roc_train_max_all += model3_roc_train_max
        final_model3_roc_all += final_model3_roc
        model3_auprc_max_all += final_model3_auprc

        print(i)
        temp_report = open('trian_report_model2_8.txt', 'a')
        print(file=temp_report)
        print('iterations:', i + 1, file=temp_report)
        #print(model1_roc_train_max, file=temp_report)
        print(final_model1_roc, file=temp_report)
        print(final_model1_auprc, file=temp_report)

        #print(model2_roc_train_max, file=temp_report)
        print(final_model2_roc, file=temp_report)
        print(final_model2_auprc, file=temp_report)

        #print(model3_roc_train_max, file=temp_report)
        print(final_model3_roc, file=temp_report)
        print(final_model3_auprc, file=temp_report)

        temp_report.close()
    final_report = open('final_report_model2_8.txt', 'w')
    print(model1_roc_train_max_all / (i + 1), file=final_report)
    print(final_model1_roc_all / (i + 1), file=final_report)
    print(model1_auprc_max_all / (i + 1), file=final_report)

    print(model2_roc_train_max_all / (i + 1), file=final_report)
    print(final_model2_roc_all / (i + 1), file=final_report)
    print(model2_auprc_max_all / (i + 1), file=final_report)

    print(model3_roc_train_max_all / (i + 1), file=final_report)
    print(final_model3_roc_all / (i + 1), file=final_report)
    print(model3_auprc_max_all / (i + 1), file=final_report)
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
        'sample_PRAD_label.pkl', 'sample_gene_names.pkl'
    ]
    test_number = 10
    validation_number = 10
    n, data_1_all, data_2_all, data_3_all, label_1_all, label_2_all, label_3_all, gene_names = read_data(
        names)

    whole_process(n, data_1_all, data_2_all, data_3_all, label_1_all,
                  label_2_all, label_3_all, validation_number, test_number,gene_names)
