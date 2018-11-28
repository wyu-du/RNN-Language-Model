# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:23:11 2018

@author: Wanyu Du
"""

import torch
import numpy as np
import os
from simple_rnnlm import SimpleRNN, get_sent_id, get_sent_tensor
import argparse


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def load_model(checkpoint_name, hidden_size):
    # load model weights from checkpoints
    load_filename = os.path.join('checkpoints', '{}.tar'.format(checkpoint_name))
    checkpoint = torch.load(load_filename, map_location='cpu')
    rnn_sd = checkpoint['rnn']
    vocab_dict = checkpoint['vocab_dict']
    
    # build model
    vocab_size = len(vocab_dict.keys())
    rnn = SimpleRNN(vocab_size, hidden_size)
    rnn.load_state_dict(rnn_sd)
    rnn = rnn.to(device)
    
    return rnn, vocab_dict
    

def get_logprob(file_name, output_fname, rnn, vocab_dict):
    rnn.eval()
    
    # load data
    sents_id = get_sent_id(file_name, vocab_dict)
    
    outs = open(output_fname, 'w', encoding='utf8')
    for i in range(len(sents_id)):
        input_variables = get_sent_tensor(sents_id[i])
        # create initial hidden_state
        hidden_state = (torch.zeros(1, 1, rnn.hidden_size).to(device), 
                        torch.zeros(1, 1, rnn.hidden_size).to(device))
        pred_sent = ''
        # forward batch of tokens through rnn one time step at a time
        for t in range(input_variables.size()[0]-1):
            output, hidden_state = rnn(input_variables[t], hidden_state)
            prob = torch.gather(output, 1, input_variables[t+1].view(1,-1))
            prob = prob.cpu().detach().numpy()[0]
            pred_sent += str(prob[0])+'\t'
        outs.write(pred_sent+'\n')
    outs.close()


def compute_perplexity(corpus_name):
    with open(corpus_name, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    print('Read {!s} sentences.'.format(len(lines)))
    n_total = 0.
    p_total = 0.
    for line in lines:
        tokens = line.strip().split('\t')
        n_total += len(tokens)+1
        sent_p = 0.
        for i in range(len(tokens)):
            sent_p += float(tokens[i].strip())
        p_total += sent_p
    avg = p_total/n_total
    out = np.exp(-avg)
    return out


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='10000_checkpoint_model')
    parser.add_argument('--hidden_size', default=32)
    args = parser.parse_args()
    
    rnn, vocab_dict = load_model(args.name, int(args.hidden_size))
    
    get_logprob('data/trn-wiki.txt', 'data/trn-logprob.txt', rnn, vocab_dict)
    get_logprob('data/dev-wiki.txt', 'data/dev-logprob.txt', rnn, vocab_dict)
    train_perplexity = compute_perplexity('data/trn-logprob.txt')
    print('Perplexity on training set:',train_perplexity)
    dev_perplexity = compute_perplexity('data/dev-logprob.txt')
    print('Perplexity on development set:',dev_perplexity)
    
