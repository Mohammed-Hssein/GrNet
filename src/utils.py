"""
Created on Fri Jul 24 11:50:36 2020

@author: mohammed hssein
"""

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module
from classes import ReOrthMap, OrthMapLayer



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
