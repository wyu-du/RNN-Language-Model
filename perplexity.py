# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:23:11 2018

@author: Wanyu Du
"""

import torch
import numpy as np

def perplexity(preds, targets, lengths):
    n_total = 0.
    p_total = 0.
    for i, (pred, target) in enumerate(zip(preds, targets)):
        n_total += lengths[i].float()
        cross_entropy = torch.log(pred[target])
        p_total += cross_entropy.sum()
    avg = p_total/n_total
    p = torch.pow(np.exp(1), -avg)
    return p
    