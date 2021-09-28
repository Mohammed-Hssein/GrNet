"""
Created on Fri Jul 31 10:39:46 2020

@author: mohammed hssein
"""


"""
Training the network on number of epochs
"""

import numpy as np
import random
import os
import sys
print("prinring ... path")
print(sys.path)
sys.path.append(os.path.join(os.getcwd(), 'src'))
print(" \n")
print(sys.path)
from src.model import GrNetwork as grnet
from torch.autograd import Variable
import torch
import datetime
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#Hyperparameters    
batch_size = 30
lr = 1
num_epoch = 10


#Data
train_images = np.load('./data/train_classes_01.npy') 
train_labels = np.load('./data/label_classes_01.npy')
len_training = train_labels.shape[0]


#monitor and track training
hist_loss = []
hist_accuracy = []
time_per_epoch = []
time_per_iter = []

#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
model = grnet(num_classes=2, num_filters=16)
#model = torch.load('./tmp/mnist/grnet_10c.model')
#model.to(device)

for epoch in range(num_epoch):
    shuffled_index = list(range(train_labels.shape[0]))
    random.seed(666)
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
        output = model(input)
        loss = F.nll_loss(output, target)
        
        # get the index of the max log-probability ==> index of right class
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).long().sum().item()
        hist_accuracy.append((correct/batch_size)*100)
        
        loss.backward()
        model.update_params(lr)
        etime = datetime.datetime.now()
        dtime = etime.second - stime.second
        #Try to adjust if time is in milliseconds .... 
        #dtime = etime.second + etime.microsecond*(10**(-6)) - (stime.second + stime.microsecond*(10**(-6)))
        time_per_iter.append(dtime)
        hist_loss.append(loss.item())
        
        print('[epoch %d/%d] [iter %d/%d] loss %f acc %f %f/batch' % (epoch, num_epoch,
                            idx_train, len_training // batch_size, loss.item(),
                            (correct / batch_size)*100, dtime))

    #Epoch finished
    time_per_epoch.append(time_per_iter)
    if not os.path.exists('./tmp/mnist'):
        os.makedirs('./tmp/mnist')
    
    plt.plot(list(range(len(hist_loss))), hist_loss, label="Loss")
    plt.plot(list(range(len(hist_accuracy))), hist_accuracy, label="Accuracy training data")
    plt.legend(loc='upper left')
    torch.save(model, './tmp/mnist/grnet_' + str(epoch + 1) + 'c.model')
    plt.savefig('./tmp/mnist/loss_c.jpg')
    plt.close()