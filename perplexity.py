# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 11:23:11 2018

@author: Wanyu Du
"""

import torch
import numpy as np
import os
from stackedlstm_rnnlm import StackedRNN, get_sent_id, get_sent_tensor


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def perplexity(corpus_name, preds, targets):
    n_total = 0.
    p_total = 0.
    for pred, target in zip(preds, targets):
        n_total += len(pred)+1
        sent_p = 0.
        for i in range(len(pred)-1):
            sent_p += pred[i][0][target[i+1]]
        p_total += sent_p
    avg = p_total/n_total
    out = np.exp(-avg)
    return out


def load_model(checkpoint_name, hidden_size):
    # load model weights from checkpoints
    load_filename = os.path.join('checkpoints', '{}.tar'.format(checkpoint_name))
    checkpoint = torch.load(load_filename, map_location='cpu')
    rnn_sd = checkpoint['rnn']
    vocab_dict = checkpoint['vocab_dict']
    
    # build model
    vocab_size = len(vocab_dict.keys())
    rnn = StackedRNN(vocab_size, hidden_size, 2)
    rnn.load_state_dict(rnn_sd)
    rnn = rnn.to(device)
    
    return rnn, vocab_dict
    

def get_perplexity_number(file_name, rnn, vocab_dict):
    rnn.eval()
    
    # load data
    sents_id = get_sent_id(file_name, vocab_dict)
    pred_sents = []
    for i in range(len(sents_id)):
        input_variables = get_sent_tensor(sents_id[i])
        # create initial hidden_state
        hidden_state = (torch.zeros(1, 1, rnn.hidden_size).to(device), 
                        torch.zeros(1, 1, rnn.hidden_size).to(device))
        pred_sent = []
        # forward batch of tokens through rnn one time step at a time
        for t in range(input_variables.size()[0]-1):
            output, hidden_state = rnn(input_variables[t], hidden_state)
            pred_sent.append(output.cpu().detach().numpy())    # shape=(seq_len, vocab_size)
        pred_sents.append(pred_sent)    # shape=(total_num, seq_len-1, vocab_size)
    p = perplexity(pred_sent, sents_id)
    return p


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
        pred_sent = '<start> '
        # forward batch of tokens through rnn one time step at a time
        for t in range(input_variables.size()[0]-1):
            output, hidden_state = rnn(input_variables[t], hidden_state)
            prob = torch.gather(output, 1, input_variables[t+1].view(1,-1))
            prob = prob.cpu().detach().numpy()[0]
            word_idx = input_variables[t+1].cpu().numpy()[0]
            pred_sent += new_dict[word_idx]+'\t'+str(prob[0])+' '
        outs.write(pred_sent+'\n')
    outs.close()


if __name__=='__main__':
    rnn, vocab_dict = load_model('10000_checkpoint_stack_2', 32)
    
    get_logprob('trn-wiki.txt', 'trn-logprob-stack2.txt', rnn, vocab_dict)
    get_logprob('dev-wiki.txt', 'dev-logprob-stack2.txt', rnn, vocab_dict)
    train_perplexity = compute_perplexity('trn-logprob-stack2.txt')
    print('Perplexity on training set:',train_perplexity)
    dev_perplexity = compute_perplexity('dev-logprob-stack2.txt')
    print('Perplexity on development set:',dev_perplexity)
    
