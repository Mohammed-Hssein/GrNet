#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:50:36 2020

@author: mohammedhssein
"""


import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module

"""
====> Any tensor is of form : [batch_size, image_h, image_w]
====> designed to contain batch of inputs images

====> Improuvements : 
    Check the backward loops : maybe some matmul are still valid for 3d tensors
    and then it cans reduce the complexity from O(N*N) to O(N)
"""
class ReOrthMap(Function):
    
    @staticmethod
    def forward(self, input):
        
        Qs = torch.zeros_like(input)
        Rs = torch.zeros((input.shape[0], input.shape[2], input.shape[2]))
        Rs_inv = torch.zeros((input.shape[0], input.shape[2], input.shape[2]))
        Ss = torch.zeros((input.shape[0], input.shape[1], input.shape[1]))
        for i in range(input.shape[0]):
            Q, R = torch.qr(input[i, :, :], some=True)
            R_inv = torch.inverse(R)
            S = torch.eye(input.shape[1]) - torch.matmul(Q, Q.transpose(0, 1))
            Qs[i, :, :] = Q
            Rs[i, :, :] = R
            Rs_inv[i, :, :] = R_inv
            Ss[i, :, :] = S
        
        self.Qs = Qs
        #self.Rs = Rs
        self.Rs_inv = Rs_inv
        self.Ss = Ss
        self.save_for_backward(input)
        return Qs
    
    @staticmethod      
    def backward(self, grad_outputs):
        
        dLdQ = grad_outputs
        grad = torch.zeros_like(grad_outputs) #grad and gradout same dims
        
        for i in range(dLdQ.shape[0]):
            ele_1 = torch.matmul(self.Ss[i, :, :].transpose(0,1), dLdQ[i, :, :])
            temp = torch.matmul(self.Qs[i, :, :].transpose(0,1), dLdQ[i, :, :])
            temp_bsym = torch.tril(temp) - torch.tril(temp.transpose(0, 1))
            ele_2 = torch.matmul(self.Qs[i, :, :], temp_bsym)
            grad[i, :, :] = torch.matmul(ele_1+ele_2, self.Rs_inv[i, :, :].transpose(0,1))
        
        return grad
    

class OrthMapLayer(Function):
    
    @staticmethod
    def forward(self, input):
        #q = 10
        Us = torch.zeros_like(input)
        Sigs = torch.zeros((input.shape[0], input.shape[1]))
        Uts = torch.zeros_like(input)
        outs = torch.zeros((input.shape[0], input.shape[1], 10))
        for i in range(input.shape[0]):
            U, Sig, Ut = torch.svd(input[i, :, :])
            Us[i, :, :]=U
            Uts[i, :, :]=Ut
            Sigs[i, :]=Sig
            outs[i, :, :] = U[:, 0:10]
        
        self.Us = Us
        self.Sigs = Sigs
        self.Uts = Uts
        return outs
    
    @staticmethod
    def backward(self, grad_outputs):
        
        dLdU = grad_outputs #needs concatenation of zeros to complete [bs, d1, d1]
        Ks = torch.zeros((dLdU.shape[0], dLdU.shape[1], dLdU.shape[1]))
        grad = torch.zeros((dLdU.shape[0], dLdU.shape[1], 12))
        for i in range(dLdU.shape[0]):
            diag = self.Sigs[i, :]
            diag = diag.contiguous()
            vs_1 = diag.view([diag.shape[0], 1])
            vs_2 = diag.view([1, diag.shape[0]])
            K = 1.0 / (vs_1 - vs_2) # matrice P
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("inf")] = 0.0
            K = K.transpose(0,1)
            #temp = torch.cat((dLdU[i, :, :], torch.zeros((dLdU[i, :, :].shape[0], 
            #                                              dLdU[i, :, :].shape[0]-10))),dim=1)
            temp = dLdU[i, :, :]
            temp = torch.matmul(self.Uts[i, :, :], temp)
            temp = torch.cat((temp, torch.zeros((12, 2))), dim=1)
            temp = K*temp
            temp = torch.matmul(self.Us[i, :, :], temp)
            temp = torch.matmul(temp, self.Uts[i, :, :])
            #print(temp.shape)
            #tronquer les q colonnes seulement
            #grad[i, :, :] = temp[:, 0:10]
            grad[i, :, :] = temp
        return grad
            
        
        
def call_reorthmap(input):
    return ReOrthMap().apply(input)


def call_orthmap(input):
    return OrthMapLayer().apply(input)

"""
def retraction(W, EucGrad, lr):
    '''
    W : parameter weight
    EucGrad : Euclidean gradient
    lr : Learning_rate
    '''
    ReimGrad = call_Reimann_grad(W.transpose(0,1), EucGrad.transpose(0,1))
    ReimGrad = ReimGrad.transpose(0,1)
    return W - lr*ReimGrad
"""

def call_Reimann_grad(W, EucGrad):
    '''
    W : weight
    EucGrad : euclidean grad
    '''
    EucGradT = EucGrad.transpose()
    U, _, V = np.linalg.svd(np.dot(W, EucGradT))
    Q = np.dot(V, U.transpose())
    Rgrad = np.dot(EucGradT, Q) - W.transpose()
    return Rgrad

def update_params_model(W, EucGrad, lr):
    '''
    to call directly when got EucGrad
    performs the update of weights W giving Euclidean gradients
    '''
    ReimGrad = call_Reimann_grad(W, EucGrad)
    ReimGrad = ReimGrad.transpose()
    return W - lr*ReimGrad

def sum_tensors(list_samples):
    '''
    '''
    n = len(list_samples)
    X = list_samples[0]
    for i in range(1, n):
        X = torch.add(X, list_samples[i])
    return (1/n)*X

"""
tensor = np.load('./data/grassmann/X_grassmann_test.npy')[0].astype(np.float32)

tensor = torch.from_numpy(tensor)

tensor = tensor.unsqueeze(0)

tensor.shape
Out[18]: torch.Size([1, 28, 10])

inp = tensor[:, 0:12, :]
"""
