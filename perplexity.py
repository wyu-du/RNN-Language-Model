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
        seq_len = input_variables.size()[0]
        output, hidden_state = rnn(input_variables[:seq_len-1], hidden_state)
        prob = torch.gather(output, 1, input_variables[1:seq_len])
        prob = prob.cpu().detach().numpy()[:, 0]
        outs.write('\t'.join(prob)+'\n')
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


def get_tst_logprob(file_name, output_fname, rnn, vocab_dict):
    rnn.eval()
    
    # load data
    sents_id = get_sent_id(file_name, vocab_dict)
    
    outs = open(output_fname, 'w', encoding='utf8')
    new_dict = {v : k for k, v in vocab_dict.items()}
    for i in range(len(sents_id)):
        input_variables = get_sent_tensor(sents_id[i])
        # create initial hidden_state
        hidden_state = (torch.zeros(1, 1, rnn.hidden_size).to(device), 
                        torch.zeros(1, 1, rnn.hidden_size).to(device))
        seq_len = input_variables.size()[0]
        output, hidden_state = rnn(input_variables[:seq_len-1], hidden_state)
        prob = torch.gather(output, 1, input_variables[1:seq_len])
        prob = prob.cpu().detach().numpy()[:, 0]
        word_idx = input_variables[1:seq_len].squeeze(1).cpu().numpy()
        for i in range(len(seq_len)):
            outs.write(new_dict[word_idx[i]]+'\t'+str(prob[i])+'\n')
    outs.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='10_checkpoint_simple')
    parser.add_argument('--hidden_size', default=32)
    args = parser.parse_args()
    
    rnn, vocab_dict = load_model(args.name, int(args.hidden_size))
    
    get_logprob('data/trn-wiki.txt', 'data/trn-logprob.txt', rnn, vocab_dict)
    get_logprob('data/dev-wiki.txt', 'data/dev-logprob.txt', rnn, vocab_dict)
    train_perplexity = compute_perplexity('data/trn-logprob.txt')
    print('Perplexity on training set:',train_perplexity)
    dev_perplexity = compute_perplexity('data/dev-logprob.txt')
    print('Perplexity on development set:',dev_perplexity)
    
#    get_tst_logprob('data/tst-wiki.txt', 'wd5jq-tst-logprob.txt', rnn, vocab_dict)