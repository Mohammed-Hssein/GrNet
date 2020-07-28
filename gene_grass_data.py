#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:58:47 2020

@author: mohammedhssein
"""

import numpy as np
#from grnet_utils import U_svd

data_train = np.load(file='./data/euclidean/X_train.npy')
data_test = np.load(file='./data/euclidean/X_test.npy')


n = data_train.shape[0] #samples
m = data_test.shape[0] #test samples
h = data_train.shape[1]
w = 10

grassmann_train = np.empty(shape=(n, h, w))
grassmann_test = np.empty(shape = (m, h, w))
"""
get U from svd decomposition
for i in range(n):
    grassmann_train[i] = U_svd(matrix=data_train[i])
    if i<m : grassmann_test[i] = U_svd(matrix=data_test[i])
"""

np.save(file='./data/grassmann/X_grassmann_train', arr=grassmann_train)
np.save(file='./data/grassmann/X_grassmann_test', arr=grassmann_test)

#----------------------------------------------------------------------------#
"""
FUll rank matrices 
saved at /data/weights/W
"""
import torch

filter_weights = []
for i in range(16):
    filter_weights.append(
        torch.nn.init.orthogonal_(
            torch.empty(12, 28)
            )
        )
# --------------------------------------------------------------------------#
"""
data_labels_training
data_labels_testing
"""

data_train_y = np.load(file='./data/euclidean/y_train.npy')
data_test_y = np.load(file='./data/euclidean/y_test.npy')

train_labels = []
test_labels = []

for i in range(len(data_train_y)):
    for j in range(10):
        if data_train_y[i][j] == 1 : train_labels.append(j)

for i in range(len(data_test_y)):
    for j in range(10):
        if data_test_y[i][j] == 1 : test_labels.append(j)

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

np.save(file='./data/grassmann/train_labels', arr=train_labels)
np.save(file='./data/grassmann/test_labels', arr=test_labels)

