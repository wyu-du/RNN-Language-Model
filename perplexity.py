# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:23:11 2018

@author: Wanyu Du
"""

import numpy as np

def perplexity(preds, targets):
    n_total = 0.
    p_total = 0.
    for pred, target in zip(preds, targets):
        n_total += len(pred)+1
        sent_p = 0.
        for i in range(len(pred)-1):
            p = -np.log(pred[i][target[i+1]])
            sent_p += p
        p_total += sent_p
    avg = p_total/n_total
    out = np.exp(-avg)
    return out
    