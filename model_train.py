#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 13:46:03 2020

@author: mohammedhssein
"""


"""
Training the network on number of epochs
"""

import numpy as np
import random
import os
import grnet_model as model
from torch.autograd import Variable
import torch
import datetime
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


batch_size = 30
lr = 0.01
num_epoch = 10

train_images = np.load('./data/grassmann/X_grassmann_train.npy') 
train_labels = np.load('./data/grassmann/train_labels.npy')
len_training = train_labels.shape[0]


model = model.GrNetwork()
hist_loss = []
time_per_epoch = []
time_per_iter = []
for epoch in range(num_epoch):
    shuffled_index = list(range(60000))
    random.seed(6666)
    random.shuffle(shuffled_index) #L list of shuffled index now is suffled_index
    
    for idx_train in range(0, len_training//batch_size):
        idx = idx_train
        shuffled_now = shuffled_index[idx*batch_size:(idx + 1)*batch_size] # L_b
        batch_label_data = []
        batch_train_data = np.zeros((batch_size, 28, 10), dtype='float32')
        i = 0
        for index in shuffled_now:
            batch_label_data.append(train_labels[index])
            batch_train_data[i, :, :] = train_images[index, :, :]
            i+=1
        input = Variable(torch.from_numpy(batch_train_data))
        target = Variable(torch.LongTensor(batch_label_data))
        
        stime = datetime.datetime.now()
        logits = model(input)
        output = F.log_softmax(logits, dim=-1)
        loss = F.nll_loss(output, target)
        
        # get the index of the max log-probability ==> index of right class
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().sum()
        
        loss.backward()
        model.update_params(lr)
        etime = datetime.datetime.now()
        dtime = etime.second - stime.second
        time_per_iter.append(dtime)
        hist_loss.append(loss.item())
        
        print('[epoch %d/%d] [iter %d/%d] loss %f acc %f %f/batch' % (epoch, num_epoch,
                            idx_train, len_training // batch_size, loss.item(),
                            correct / batch_size, dtime))
    #Epoch finished
    time_per_epoch.append(time_per_iter)
    if not os.path.exists('./tmp/mnist'):
        os.makedirs('./tmp/mnist')
    plt.plot(list(range(len(hist_loss))), hist_loss)
    torch.save(model, './tmp/mnist/grnet_' + str(epoch + 1) + 'c.model')
    plt.savefig('./tmp/mnist/loss_c.jpg')
    plt.close()
    

"""
model = model.GrNet()
hist_loss=[]

for epoch :
    
    get list of shuffled indexes L = [1000, 4, 5, 599, ....]
    
    for idx_train in range(0, len(train)//batch_size):
        idx = ids_train
        L_b = L[idx*batch_size, (idx+1)*batch_size]
        
        batch_label_data = []
        batch_train_data = np.zeros(batch, 28, 10)
        i = 0
        for x in L_b :
            batch_label_data.append(data_y[x]) elem correspo à l'indice x
            batch_train_data[i, :, :] = data_X[x, :, :]
            i+=1
        
        input = Variable(from_np(batch_train_data))
        target = Variable(torch.LongTensor(batch_label_data))
        
        stime = datetime.datetime.now()
        logits = model(input)
        output = log_sfotmax(logits, dim=-1)
        loss = nll_loss(output, target)
        
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        
        correct = pred.eq(target.data.view_as(pred)).long().sum()
        
        loss.backward()
        model.update_para(lr)
        etime = datetime.datetime.now()
        dtime = etime.second - stime.second
        
        hist_loss.append(loss.data[0])
        
        print("")
        
    #one epoch done
    
    if not os.path.exists('./tmp/mnist'):
        os.makedirs('./tmp/mnist')
    plt.plot(list(range(len(hist_loss))), hist_loss)
    torch.save(model, './tmp/mnist/grnet_' + str(epoch + 1) + 'c.model')
    plt.savefig('./tmp/mnist/loss_c.jpg')
    plt.close()
"""

