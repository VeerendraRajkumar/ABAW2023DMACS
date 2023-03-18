'''
Aum Sri Sai Ram

ABAW-5 for Expr, , 2023

losses
 
'''

from __future__ import print_function

import torch.nn.functional as F


def cross_entropy_loss_without_weights(y_hat, y):
    return F.cross_entropy(y_hat, y, reduction='none')
