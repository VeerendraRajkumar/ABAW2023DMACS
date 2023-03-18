'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

MutexMatch with threshold - mutexmatch utils
code adapted from https://github.com/NJUyued/MutexMatch4SSL 
'''

import torch
import numpy as np
import torch.nn.functional as F
from train_utils import ce_loss

class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value

def consistency_loss(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, k, name='ce', T=1.0, p_cutoff=[0.0]*7, use_hard_labels=True):
#                       Waug -ve logits,           Saug -ve logits,  Waug +ve , Saug +ve,|
#                                                       intensity of consistency for Ln <-

    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1) # p^w
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask, mask_dis = mask_generate(max_probs, max_idx, logits_w.shape[0], p_cutoff) # @V[15 Feb 23]

        pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1) # q^w

        if k==0 or k==pseudo_label.size(1):
            pass
        else:
            filter_value = float(0)
            indices_to_remove = pseudo_label_reverse < torch.topk(pseudo_label_reverse, k)[0][..., -1, None] # g(i)
            pseudo_label_reverse[indices_to_remove] = filter_value # setting the value for indices to remove as 0 
            # i.e., removing the values that are to be removed 
            logits_x_ulb_s_reverse[indices_to_remove] = filter_value # setting the value for indices to remove as 0 

        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels = False, reduction='none') * mask_dis
        else:
            # 
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask

        return masked_loss.mean(), masked_reverse_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def mask_generate(max_probs, max_idx, batchsize, threshold):
    mask = torch.zeros(batchsize)
    mask_dis = torch.zeros(batchsize)
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        md = max_probs[idx].lt(threshold[i]).float()
        for k in range(len(idx)):
            mask[idx[k]]+=m[k].cpu()
            mask_dis[idx[k]]+=md[k].cpu()
    return mask.cuda(), mask_dis.cuda()

def mask_generate_greaternequal(max_probs, max_idx, batchsize, threshold):
    mask = torch.zeros(batchsize)
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask[idx[k]]+=m[k].cpu()
    return mask.cuda()

def mask_generate_less(max_probs, max_idx, batchsize, threshold):
    mask_ori = torch.zeros(batchsize)
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].lt(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k].cpu()
    return mask_ori.cuda()