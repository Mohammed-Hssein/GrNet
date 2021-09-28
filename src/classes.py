#File containing all the classes for the feedforward pass.

import torch
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module

"""
> Any tensor is of form : [batch_size, image_h, image_w]
> designed to contain batch of inputs images
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
            
        