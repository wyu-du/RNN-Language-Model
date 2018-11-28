# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:40:16 2018

@author: Wanyu Du
"""

import torch
from torch import nn
import os
import numpy as np
import itertools
import argparse


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.read().strip().split('\n')
    print('Read {!s} sentences.'.format(len(lines)))
    words = []    
    max_len = 0
    for line in lines:
        if len(line.split())>max_len:
            max_len = len(line.split(' '))
        for word in line.split():
            words.append(word)
    print('Counted tokens:', len(words))
    return words, max_len            


def build_vocab(words, vocab_dict):
    for word in words:
        if word not in vocab_dict.keys():
            vocab_dict[word] = len(vocab_dict.keys())
    print('Vocabulary size:', len(vocab_dict.keys()))
    return vocab_dict


def get_sent_id(file_path, vocab_dict):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    sents = []    
    for line in lines:
        sent = []
        for word in line.split():
            if word in vocab_dict.keys():
                sent.append(vocab_dict[word])
            else:
                sent.append(vocab_dict['<unk>'])
        sents.append(sent)
    return sents


def get_sent_tensor(sents_id, mini_batch):
    batch_idx = np.random.randint(low=0, high=len(sents_id), size=mini_batch)
    sents_batch = sents_id[batch_idx]
    sents_batch = list(itertools.zip_longest(*sents_batch, fillvalue=0))
    sent = torch.LongTensor(sents_id)
    sent = sent.t()   # shape=(max_len, batch_size)
    print(sent.size()) 
    sent = sent.to(device)
    return sent
    

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.decode = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_token, hidden_state):
        '''
        input_token: (1, batch_size)
        hidden_state: (2, batch_size, hidden_size)
        '''
        embedded = self.embedding(input_token)         # (1, batch_size, hidden_size)
        # hidden_state: (2, batch_size, hidden_size)
        # output: (1, batch_size, hidden_size)
        output, hidden = self.lstm(embedded, hidden_state) 
        out = self.decode(hidden[0])             # (1, batch_size, vocab_size)
        out = out.squeeze(0)
        out = nn.functional.log_softmax(out, dim=1)    # (batch_size, vocab_size)
        return out, hidden


def train(input_variables, mini_batch, loss_fun, rnn, rnn_optimizer, clip):
    rnn_optimizer.zero_grad()
    
    input_variables = input_variables.to(device)
    
    # create initial hidden_state
    hidden_state = (torch.zeros(1, mini_batch, rnn.hidden_size).to(device),  # h_0
                    torch.zeros(1, mini_batch, rnn.hidden_size).to(device))  # c_0
    
    loss = 0.
    printed_loss = []
    num_tokens = 0.
    
    # forward batch of tokens through rnn one time step at a time
    for t in range(input_variables.size()[0]-1):
        # output: (batch_size, vocab_size)
        # hidden_state: (2, batch_size, hidden_size)
        output, hidden_state = rnn(input_variables[t], hidden_state)
        token_loss = loss_fun(output, input_variables[t+1]) 
        loss += token_loss
        num_tokens += 1
        printed_loss.append(loss.item())
    
    # perform backpropagation
    loss.backward()
    
    # clip gradients
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    
    # adjust model weights
    rnn_optimizer.step()
    
    return sum(printed_loss)/num_tokens


def train_iters(iters, mini_batch, vocab_dict, hidden_size, lr, norm_clipping):
    # load data
    sents_id = get_sent_id('trn-wiki.txt', vocab_dict)
    
    # build model
    rnn = SimpleRNN(vocab_size=len(vocab_dict.keys()), hidden_size=hidden_size)
    rnn.to(device)
    rnn.train()
    
    # initialize optimizer
    rnn_optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    # initialize loss function
    loss_fun = nn.NLLLoss()
    
    # start training
    for k in range(iters):
        input_variables = get_sent_tensor(sents_id, mini_batch)
        loss = train(input_variables, loss_fun, rnn, rnn_optimizer, norm_clipping)
        print("Iteration: {}; Average loss: {:.4f}".format(k, loss))
        
    # Save checkpoint
    directory = 'checkpoints'
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save({
        'rnn': rnn.state_dict(),
        'vocab_dict': vocab_dict,
    }, os.path.join(directory, '{}_{}.tar'.format((k+1), 'checkpoint_simple')))
    
    return rnn



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', default=10000)
    parser.add_argument('--mini_batch', default=32)
    args = parser.parse_args()
    
    vocab_dict = {'<padding>':0, '<unk>':1, '<num>':2, '<start>':3, '<stop>':4}
    hidden_size = 32
    lr = 0.001
    norm_clipping = 10
    
    trn_words, max_len = load_data('trn-wiki.txt')
    vocab_dict = build_vocab(trn_words, vocab_dict)
    
    rnn = train_iters(int(args.iters), int(args.mini_batch), vocab_dict, hidden_size, lr, norm_clipping)
