'''
Aum Sri Sai Ram

ABAW-5 for Expr, , 2023

methods to support eval
 
'''
from __future__ import print_function, absolute_import

import numpy as np
from sklearn.metrics import f1_score


__all__ = ['accuracy','EXPR_metric']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def EXPR_metric(x, y): 
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)

    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)

    f1 = f1_score(x, y, average= 'macro')
    return f1