"""
Created on Mon Aug  3 12:48:58 2020

@author: mohammed hssein
"""

import torch
import  torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from classes import ReOrthMap, OrthMapLayer
from utils import *


use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
tenType = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class GrNetwork(torch.nn.Module):
    
    def __init__(self, num_classes, num_filters):
        super(GrNetwork, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.filter_weights = []
        self.layer_wieights = []

        for i in range(0, self.num_filters, 1):
            self.filter_weights.append(torch.load("./data/weights/W")[i].type(tenType))
            self.filter_weights[i] = Variable(self.filter_weights[i], requires_grad=True)
        """
		#This was made for other pusposes
        if self.num_classes == 10:
            self.fc_w = Variable(torch.load('./data/weights/FC'), requires_grad = True)
        if self.num_classes == 3:
            self.fc_w = Variable(torch.load('./data/weights/FC_3'), requires_grad = True)
        if self.num_classes == 2:
            self.fc_w = Variable(torch.load('./data/weights/FC_2'), requires_grad = True)

        self.bias = Variable(torch.randn((1, 2)), requires_grad = True)
        #self.bias = Variable(torch.load('./data/weights/bias'), requires_grad = True)
        """
        self.fc_w1 = Variable(torch.randn(144, 10), requires_grad = True)
        self.fc_b1 = Variable(torch.randn(1, 10), requires_grad = True)

        self.fc_w2 = Variable(torch.randn(72, 2), requires_grad = True)
        self.fc_b2 = Variable(torch.randn(1, 2), requires_grad = True)
        
        self.fc_w3 = Variable(torch.randn(36, 12), requires_grad = True)
        self.fc_b3 = Variable(torch.randn(1, 12), requires_grad = True)
        
        self.fc_w4 = Variable(torch.randn(12, 2), requires_grad = True)
        self.fc_b4 = Variable(torch.randn(1, 2), requires_grad = True)
        
    def forward(self, input):
        '''
        forward pass 1 block
        '''
        input = input.type(tenType)
        batch_size = input.shape[0]
        #W1_c = self.W_1.contiguous()
        W1_c = []
        W1 = []
        X1 = []
        for i in range(self.num_filters):
            W1_c.append(self.filter_weights[i].contiguous())
            W1.append(W1_c[i].view([1, W1_c[i].shape[0], W1_c[i].shape[1]]))
            X1.append(torch.matmul(W1[i], input))
        
        X2 = [call_reorthmap(x) for x in X1]
        X3 = [torch.matmul(x, x.transpose(1, 2)) for x in X2]
        X4 = sum_tensors(list_samples=X3)
        X5 = call_orthmap(X4)
        X6 = torch.matmul(X5, X5.transpose(1, 2))
        FC = X6.view([batch_size, -1])
        logits = torch.add(torch.matmul(FC, self.fc_w1), self.fc_b1)
        output = F.log_softmax(logits, dim=-1)
        return output
    
    def update_params(self, lr):
        '''
        lr : learning rate
        '''
        #Compute gradients manually with the our classes
        eugrad_W1 = [self.filter_weights[i].grad.data.numpy() for i in range(self.num_filters)]
        W1_np = [self.filter_weights[i].data.numpy() for i in range(self.num_filters)]
       
        new_W1 = []
        for i in range(self.num_filters):
            #new_W1.append(utils.update_params_model(W1_np[i], eugrad_W1[i],lr))
            new_W1.append(update_params_model(W1_np[i], eugrad_W1[i],lr))
            
        #update the weights
        for i in range(self.num_filters):
            #self.W_1.data.copy_(tenType(new_W1))
            self.filter_weights[i].data.copy_(tenType(new_W1[i]))
            #self.filter_weights[i].data.copy_(torch.FloatTensor(new_W1[i]))

		#Update wieghts
        with torch.no_grad():
            self.fc_w1.data -= lr * self.fc_w1.grad.data
            self.fc_b1 -= lr*self.fc_b1.grad.data        
        
        #set gradients to zero manually
        for i in range(self.num_filters):
            self.filter_weights[i].grad.data.zero_()
        #self.W_1.grad.data.zero_()
        self.fc_w1.grad.data.zero_()
        self.fc_b1.grad.data.zero_()
        
