#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:45:05 2020

@author: mohammedhssein
"""


import torch
from torch.autograd import Variable
import numpy as np
import utils_model as utils


class GrNetwork(torch.nn.Module):
    
    def __init__(self):
        super(GrNetwork, self).__init__()
        self.num_filters = 16
        """
        initialize for once full rank matrices and load them for ones
        ===> reduces complexity
        
        Start with one filter weight for testing the entire architechture and 
        then generalize for 16
        
        d1 = 12, d0 = 28, q = 10
        """
        self.filter_weights = []
        for i in range(16):
            self.filter_weights.append(torch.load('./data/weights/W')[i])
            self.filter_weights[i] = Variable(self.filter_weights[i], requires_grad=True)
        #self.W_1 = Variable(self.filter_weights[0], requires_grad = True)
        #W_1 = filter_weights[0]
        self.fc_w = Variable(torch.randn((144, 10)) , requires_grad = True)
    
    def forward(self, input):
        '''
        forward pass 1 block
        '''
        batch_size = input.shape[0]
        #W1_c = self.W_1.contiguous()
        W1_c = []
        W1 = []
        X1 = []
        for i in range(16):
            W1_c.append(self.filter_weights[i].contiguous())
            W1.append(W1_c[i].view([1, W1_c[i].shape[0], W1_c[i].shape[1]]))
            X1.append(torch.matmul(W1[i], input))
        
        #X1 = torch.matmul(W1, input)
        #X2 = utils.call_reorthmap(X1)
        X2 = [utils.call_reorthmap(x) for x in X1]
        
        #X3 = torch.matmul(X2, X2.transpose(1, 2)) #list of X3_i
        X3 = [torch.matmul(x, x.transpose(1, 2)) for x in X2]
        #X4 = X3 #mean of X3_i
        X4 = utils.sum_tensors(list_samples=X3)
        
        X5 = utils.call_orthmap(X4)
        X6 = torch.matmul(X5, X5.transpose(1, 2))
        
        FC = X6.view([batch_size, -1])
        out = torch.matmul(FC, self.fc_w)
        
        return out
    
    def update_params(self, lr):
        '''
        lr : learning rate
        '''
        #get weights and euclidean gradients in np forms
        #eugrad_W1 = self.W_1.grad.data.numpy()
        eugrad_W1 = [self.filter_weights[i].grad.data.numpy() for i in range(16)]
        #W1_np = self.W_1.data.numpy()
        W1_np = [self.filter_weights[i].data.numpy() for i in range(16)]
        
        #new weihgts updated
        #new_W1 = utils.update_params_model(W1_np, eugrad_W1,lr)
        new_W1 = []
        for i in range(16):
            new_W1.append(utils.update_params_model(W1_np[i], eugrad_W1[i],lr))
            
        #update the weights
        for i in range(16):
            #self.W_1.data.copy_(torch.DoubleTensor(new_W1))
            self.filter_weights[i].data.copy_(torch.DoubleTensor(new_W1[i]))
        self.fc_w.data -= lr * self.fc_w.grad.data
        
        #set gradients to zero manually
        for i in range(16):
            self.filter_weights[i].grad.data.zero_()
        #self.W_1.grad.data.zero_()
        self.fc_w.grad.data.zero_()

    
"""
tensor = np.load('./data/grassmann/X_grassmann_test.npy')[0].astype(np.float32)

tensor = torch.from_numpy(tensor)

tensor = tensor.unsqueeze(0)

net = GrNetwork()

output = net.forward(tensor)

target = output.clone().detach()

target[0][1] = 3

loss = torch.nn.functional.mse_loss(output, target)

"""
    


