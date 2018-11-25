# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:40:16 2018

@author: Wanyu Du
"""

import torch
from torch import nn
import itertools


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
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


def zeroPadding(sents, fillvalue=0):
    padded_sents = itertools.zip_longest(*sents, fillvalue=fillvalue)
    return list(padded_sents)


def getSentId(file_path, vocab_dict):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    print('Read {!s} sentences.'.format(len(lines)))
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
    

def getBatchSents(sents_id):
    sent_lengths = []
    padded_sents = zeroPadding(sents_id)
    for sent_id in sents_id:
        sent_lengths.append(len(sent_id))
    padded_sents = torch.LongTensor(padded_sents)
    padded_sents = padded_sents.unsqueeze(1)
    padded_sents = padded_sents.to(device)    # (max_len, 1, batch_size)
    sent_lengths = torch.LongTensor(sent_lengths)
    sent_lengths = sent_lengths.to(device)    # (batch_size)
    return padded_sents, sent_lengths


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
        hidden_state: (num_layers*num_directions, batch_size, hidden_size)
        '''
        embedded = self.embedding(input_token)     # (1, batch_size, hidden_size)
        # hidden_state: (1, batch_size, hidden_size)
        # output: (1, batch_size, hidden_size)
        output, hidden = self.lstm(embedded, hidden_state) 
        out = self.decode(hidden[0]+hidden[1])     # (1, batch_size, vocab_size)
        out = out.squeeze(0)
        out = nn.functional.softmax(out, dim=1)    # (batch_size, vocab_size)
        return out, hidden


def train(input_variables, lengths, loss_fun, rnn, rnn_optimizer, max_len, batch_size, clip):
    rnn_optimizer.zero_grad()
    
    input_variables = input_variables.to(device)
    
    # create initial hidden_state
    hidden_state = (torch.zeros(1, batch_size, rnn.hidden_size), 
                    torch.zeros(1, batch_size, rnn.hidden_size))
    
    loss = 0.
    num_tokens = 0.
    # forward batch of tokens through rnn one time step at a time
    for t in range(max(lengths)-1):
        output, hidden_state = rnn(input_variables[t], hidden_state)
        _, topi = output.topk(1)
        token_loss = loss_fun(rnn.embedding(topi.view(1, -1)), rnn.embedding(input_variables[t+1]))  
        loss += token_loss
        num_tokens += 1
    
    # perform backpropagation
    loss.backward()
    
    # clip gradients
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    
    # adjust model weights
    rnn_optimizer.step()
    
    return loss/num_tokens


if __name__=='__main__':
    vocab_dict = {'<pad>':0, '<unk>':1, '<num>':2, '<start>':3, '<stop>':4}
    batch_size = 1
    lr = 0.001
    norm_clipping = 5.0
    
    trn_words, max_len = load_data('trn-wiki.txt')
    vocab_dict = build_vocab(trn_words, vocab_dict)
    
    # build model
    rnn = SimpleRNN(vocab_size=len(vocab_dict.keys()), hidden_size=32)
    rnn.to(device)
    rnn.train()
    # initialize optimizer
    rnn_optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    loss_fun = nn.MSELoss()
    
    # start training
    for i in range(10):
        # load data
        sents_id = getSentId('trn-wiki.txt', vocab_dict)
        avg_loss = 0.
        for i in range(len(sents_id)//batch_size-1):
            sents_id_batch = sents_id[i*batch_size:(i+1)*batch_size]
            input_variables, lengths = getBatchSents(sents_id_batch)
            loss = train(input_variables, lengths, loss_fun, rnn, rnn_optimizer, 
                     max_len, batch_size, norm_clipping)
            avg_loss += loss
        print("Iteration: {}; Average loss: {:.4f}".format(i, avg_loss/len(sents_id)))
        