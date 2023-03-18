'''
Aum Sri Sai Ram

ABAW-5 for Expr, 2023

Noise aware model - losses

'''

import torch 
import torch.nn as nn
import torch.nn.functional as F

eps = 1e-8


class DCE(nn.Module):
    """
    Implementing Noise Robust DCE Loss 
    Details: take smooth labels (generated here) . Generates the average of all the predictions and do class wise sum
    If the prediction probability of ground truth is greater than the average for that class then it's clean.
    Drive only clean samples towards celoss
    """

    def __init__(self,  num_class=7,epsilon=0.35, reduction="mean"):
        super(DCE, self).__init__()
        
        self.reduction = reduction
        self.num_class = num_class
        self.epsilon = epsilon

    def forward(self, prediction, target_label,clean=False, one_hot=True):

        if one_hot:
            y_true = F.one_hot(target_label.type(torch.LongTensor), num_classes=self.num_class).float().cuda()
        y_pred = F.softmax(prediction, dim=1)
        y_pred = torch.clamp(y_pred, eps, 1-eps)
        
        N,C = y_pred.shape
        smooth_labels = torch.full(size=(N, C), fill_value=self.epsilon/(C - 1)).cuda()
        smooth_labels.scatter_(dim=1, index=torch.unsqueeze(target_label, dim=1), value=1-self.epsilon)
        
        pred_tmp = torch.sum(y_true * y_pred, axis=-1).reshape(-1, 1)
        avg = torch.mean(y_pred, dim=0)
        avg = avg.reshape(-1, 1)
        avg_ref = torch.matmul(y_true.type(torch.float), avg)
        pred = torch.where((pred_tmp >= avg_ref ), pred_tmp, torch.zeros_like(pred_tmp)) #confident
        
        # confident_idx will tell us which examples are 'trustworthy' for the given batch
        confident_idx = torch.where(pred != 0.)[0]
        
        if len(confident_idx) != 0:
            prun_targets = smooth_labels[confident_idx]#torch.argmax(torch.index_select(smooth_labels, 0, confident_idx), dim=1)
            weighted_loss = cross_entropy(torch.index_select(prediction, 0, confident_idx),prun_targets, reduction=self.reduction)
        else:
            weighted_loss = cross_entropy(prediction, smooth_labels)

        return weighted_loss



        
def cross_entropy(logits, labels, reduction='mean'):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)
	
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / logits.size(0)
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')        
        
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)

def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)
