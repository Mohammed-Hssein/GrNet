#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 20:50:02 2020

@author: mohammedhssein
"""


"""
testing the model
"""

import numpy as np
import random
import os
import grnet_model_plus_linear as grnet
from torch.autograd import Variable
import torch
import datetime
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


test_images = np.load('./data/grassmann/X_grassmann_test.npy') 
test_labels = np.load('./data/grassmann/test_labels.npy')

model = torch.load('./results/mnist_01/models/grnet_3_FC_layers.model')
"""
#Un comment this part tu ose the params obtained for the best model. Change path to load the 
#right model.
"""
batch_size = 30
num_epoch = 100
len_testing = test_labels.shape[0]

hist_accuracy = []

for epoch in range(num_epoch):
    shuffled_index = list(range(test_labels.shape[0]))
    random.seed(6666)
    random.shuffle(shuffled_index) #L list of shuffled index now is suffled_index
    
    for idx_test in range(0, len_testing//batch_size):
        idx = idx_test
        shuffled_now = shuffled_index[idx*batch_size:(idx + 1)*batch_size] # L_b
        batch_label_data = []
        batch_test_data = np.zeros((batch_size, 28, 10), dtype='float32')
        i = 0
        for index in shuffled_now:
            batch_label_data.append(test_labels[index])
            batch_test_data[i, :, :] = test_images[index, :, :]
            i+=1
        input = torch.Tensor(torch.from_numpy(batch_test_data))
        target = torch.from_numpy(np.array(batch_label_data)).type(torch.LongTensor)
        output = model(input)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().sum().item()
        hist_accuracy.append((correct/batch_size)*100)
        
        print('[epoch %d/%d] [iter %d/%d] acc %f' % (epoch, num_epoch,
                            idx_test, len_testing // batch_size, 
                            (correct / batch_size)*100))
    
    #epoch finished
    if not os.path.exists('./tmp/mnist/test'):
        os.makedirs('./tmp/mnist/test')
    
    
    plt.plot(list(range(len(hist_accuracy))), hist_accuracy, label="Accuracy test data")
    plt.legend(loc='upper left')
    plt.savefig('./tmp/mnist/test/accuracy_c.jpg')
    plt.close()

