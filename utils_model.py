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

#use_cuda = torch.cuda.is_available()
#tenType = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
tenType = torch.FloatTensor
 
class ReOrthMap(Function):
    
    @staticmethod
    def forward(self, input):
        input = input.type(tenType)
        Qs = torch.zeros_like(input).type(tenType)
        Rs = torch.zeros((input.shape[0], input.shape[2], input.shape[2])).type(tenType)
        Rs_inv = torch.zeros((input.shape[0], input.shape[2], input.shape[2])).type(tenType)
        Ss = torch.zeros((input.shape[0], input.shape[1], input.shape[1])).type(tenType)
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
        #self.save_for_backward(input)
        return Qs
    
    @staticmethod      
    def backward(self, grad_outputs):
        
        dLdQ = grad_outputs.type(tenType)
        grad = torch.zeros_like(grad_outputs).type(tenType) #grad and gradout same dims
        
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
        input = input.type(tenType)
        Us = torch.zeros_like(input).type(tenType)
        Sigs = torch.zeros((input.shape[0], input.shape[1])).type(tenType)
        Uts = torch.zeros_like(input).type(tenType)
        outs = torch.zeros((input.shape[0], input.shape[1], 10)).type(tenType)
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
        
        dLdU = grad_outputs.type(tenType) #needs concatenation of zeros to complete [bs, d1, d1]
        Ks = torch.zeros((dLdU.shape[0], dLdU.shape[1], dLdU.shape[1])).type(tenType)
        grad = torch.zeros((dLdU.shape[0], dLdU.shape[1], 12)).type(tenType)
        for i in range(dLdU.shape[0]):
            diag = self.Sigs[i, :]
            diag = diag.contiguous()
            vs_1 = diag.view([diag.shape[0], 1])
            vs_2 = diag.view([1, diag.shape[0]])
            K = 1.0 / (vs_1 - vs_2) # matrice P
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("inf")] = 0.0
            K = K.transpose(0,1)
            temp = dLdU[i, :, :]
            temp = torch.matmul(self.Uts[i, :, :], temp).type(tenType)
            temp = torch.cat((temp, torch.zeros((12, 2), dtype=torch.float32)), dim=1)
            temp = K*temp
            temp = torch.matmul(self.Us[i, :, :], temp)
            temp = torch.matmul(temp, self.Uts[i, :, :])
            grad[i, :, :] = temp
        return grad
            
        
        
def call_reorthmap(input):
    return ReOrthMap().apply(input)


def call_orthmap(input):
    return OrthMapLayer().apply(input)



def call_Reimann_grad(W, EucGrad):
    """
    W : weight to be updated
    EucGrad : euclidean gradient
    """
    EucGradT = EucGrad.astype(np.double).transpose()
    W = W.astype(np.double)
    U, _, V = np.linalg.svd(np.dot(W, EucGradT))
    Q = np.dot(V, U.transpose())
    Rgrad = np.dot(EucGradT, Q) - W.transpose()
    Rgrad = Rgrad/np.linalg.norm(Rgrad)
    return Rgrad.astype(np.double)

def update_params_model(W, EucGrad, lr):
    """
    performs the update of weights W giving Euclidean gradients
    """
    ReimGrad = call_Reimann_grad(W, EucGrad)
    ReimGrad = ReimGrad.transpose()
    return W.astype(np.double) - lr*ReimGrad

def sum_tensors(list_samples):
    '''
    Function to call durin the mean pooling
    '''
    n = len(list_samples)
    X = list_samples[0]
    for i in range(1, n):
        X = torch.add(X, list_samples[i])
    return (1/n)*X


def U_svd(matrix):
    '''
    return the 10 fisrt largest eigen vectors
    '''
    U = np.linalg.svd(matrix)[0]
    return U[:, 0:10]
#--------------------------------------------------------------------#
#  use weight matrices just full row rank
#  ===> construct by taking orthogonal of emppty torch
#--------------------------------------------------------------------#

"""
NOTE:
----

    Any function with the mention v2, is another version of the original one. In general, those functions
    are for the sake of trying different retraction functions.
    
    This section of functions are for another retraction. See report 
"""
def call_Reimann_grad_v2(W, EucGrad):
    '''
    
    '''
    Wt_W = np.matmul(W.transpose(), W)
    Reim_grad = np.matmul(EucGrad, Wt_W)
    Reim_grad = EucGrad - Reim_grad
    return Reim_grad

def retraction(X):
    '''
    Projecto onto the manifold of fixed rank matrices
    Projection = sum of ui.vi.T for rank r first terms
    '''
    u, d, v = np.linalg.svd(X)
    rank = np.linalg.matrix_rank(X)
    # to chkck d[0] or abs(d[0])
    x = d[0]*np.matmul(np.expand_dims(u[:, 0], axis = 1), np.expand_dims(v[:, 0].transpose(), axis=0))
    for i in range(1, rank):
        x += d[i]*np.matmul(np.expand_dims(u[:, i], axis = 1), np.expand_dims(v[:, i].transpose(), axis=0))
    
    return x

def update_params_model_v2(W, EucGrad, lr):
    '''
    '''
    reim_grad = call_Reimann_grad_v2(W, EucGrad)
    w = retraction(W - lr*reim_grad)
    return w

#----------------------------- END ------------------------------------#