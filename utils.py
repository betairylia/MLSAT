import torch
# import ot
import numpy as np
import torch.nn.functional as F

def CrossEntropy(X, Y, eps = 1e-12):
    return - torch.sum(X * torch.log(Y + eps), dim = -1)

def JSDivergence(X, Y, eps = 1e-12):
    M = (X + Y) / 2
    JSD = 0.5 * torch.sum(X * torch.log(X / (M + eps) + eps), -1) + 0.5 * torch.sum(Y * torch.log(Y / (M + eps) + eps), -1)
    return JSD

def Softmax_KL_div0(A, B): # newly defined to compute r_vadv more precisely
    softA_log_softA_B = torch.softmax(A, 1)*(torch.log_softmax(A, 1) - torch.log_softmax(B, 1))
    loss = torch.sum(softA_log_softA_B) 
    return loss

def softmax_shannon_ent(out_values):
    bs = out_values.shape[0]
    log_soft = torch.log_softmax(out_values, 1)
    soft = torch.softmax(out_values, 1)
    loss = torch.sum(-soft*log_soft) / bs
    return loss

def autoNormalize(X):
    s = X.shape
    return F.normalize(X.view(s[0], -1), p = 2, dim = 1).view(s)
    