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
    sents_batch = []
    batch_idx = np.random.randint(low=0, high=len(sents_id), size=mini_batch)
    for idx in batch_idx:
        sents_batch.append(sents_id[idx])
    sents_batch.sort(key = lambda x: len(x), reverse=True)
    sent_len = []
    for i in range(len(sents_batch)):
        sent_len.append(len(sents_batch[i])-1)
    sents_batch = list(itertools.zip_longest(*sents_batch, fillvalue=0))
    sent = torch.LongTensor(sents_batch)  # shape=(max_len, batch_size)
    sent = sent.to(device)
    sent_len = torch.LongTensor(sent_len)
    sent_len = sent_len.to(device)
    return sent, sent_len
    

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False)
        self.decode = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_tokens, hidden_state, seq_len):
        '''
        input_token: (max_len, batch_size)
        hidden_state: (2, batch_size, hidden_size)
        '''
        embedded = self.embedding(input_tokens)        # (max_len, batch_size, hidden_size)
        # hidden_state: (2, batch_size, hidden_size)
        # output: (1, batch_size, hidden_size)
        embedded = embedded.transpose(0, 1)            # (batch_size, max_len, hidden_size)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, seq_len, batch_first=True)
        output, hidden = self.lstm(packed, hidden_state) 
        output_pack, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output_pack = output_pack.transpose(0, 1)
        out = self.decode(output_pack)                 # (max_len, batch_size, vocab_size)
        out = nn.functional.log_softmax(out, dim=2)    # (max_len, batch_size, vocab_size)
        return out, hidden


def train(input_variables, input_lengths, loss_fun, rnn, rnn_optimizer, clip, mini_batch):
    rnn_optimizer.zero_grad()
    
    input_variables = input_variables.to(device)
    
    # create initial hidden_state
    hidden_state = (torch.zeros(1, mini_batch, rnn.hidden_size).to(device),  # h_0
                    torch.zeros(1, mini_batch, rnn.hidden_size).to(device))  # c_0
    
    max_len = input_variables.size()[0]
    output, hidden_state = rnn(input_variables[:max_len-1], hidden_state, input_lengths)
    output_flatten = output.view((max_len-1)*mini_batch, output.size()[2])
    target_flatten = input_variables[1:max_len].view((max_len-1)*mini_batch, 1).squeeze(1)
    loss = loss_fun(output_flatten, target_flatten) 
    printed_loss = loss.item()
    
    # perform backpropagation
    loss.backward()
    
    # clip gradients
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    
    # adjust model weights
    rnn_optimizer.step()
    
    return printed_loss


def train_iters(iters, vocab_dict, hidden_size, lr, norm_clipping, mini_batch):
    # load data
    sents_id = get_sent_id('data/trn-wiki.txt', vocab_dict)
    
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
        avg_loss = 0
        for i in range(0, len(sents_id), mini_batch):
            input_variables, input_lengths = get_sent_tensor(sents_id, mini_batch)
            loss = train(input_variables, input_lengths, loss_fun, rnn, rnn_optimizer, norm_clipping, mini_batch)
            avg_loss += loss
        
        # Save checkpoint
        if (k+1) % 10 == 0:
            directory = 'checkpoints'
            if not os.path.exists(directory):
                os.makedirs(directory)
    
            torch.save({
                'rnn': rnn.state_dict(),
                'vocab_dict': vocab_dict,
            }, os.path.join(directory, '{}_{}.tar'.format((k+1), 'checkpoint_batch')))
    
        print("Epoch: {}; Average loss: {:.4f}".format(k, avg_loss/(len(sents_id)/mini_batch)))
    return rnn



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', default=10)
    parser.add_argument('--mini_batch', default=16)
    args = parser.parse_args()
    
    vocab_dict = {'<padding>':0, '<unk>':1, '<num>':2, '<start>':3, '<stop>':4}
    hidden_size = 32
    lr = 0.1
    norm_clipping = 0.5
    
    trn_words, max_len = load_data('data/trn-wiki.txt')
    vocab_dict = build_vocab(trn_words, vocab_dict)
    
    rnn = train_iters(int(args.iters), vocab_dict, hidden_size, lr, norm_clipping, int(args.mini_batch))
