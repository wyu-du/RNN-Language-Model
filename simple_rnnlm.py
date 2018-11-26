# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 10:40:16 2018

@author: Wanyu Du
"""

import torch
from torch import nn
import os
from perplexity import perplexity


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


def get_sent_tensor(sent_id):
    sent = torch.LongTensor([sent_id])
    sent = sent.t()
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
        input_token: (1, 1)
        hidden_state: (num_layers*num_directions, 1, hidden_size)
        '''
        embedded = self.embedding(input_token)     # (1, hidden_size)
        embedded = embedded.unsqueeze(0)
        # hidden_state: (1, 1, hidden_size)
        # output: (1, 1, hidden_size)
        output, hidden = self.lstm(embedded, hidden_state) 
        out = self.decode(hidden[0]+hidden[1])     # (1, 1, vocab_size)
        out = out.squeeze(0)
        out = nn.functional.softmax(out, dim=1)    # (1, vocab_size)
        return out, hidden


def train(input_variables, loss_fun, rnn, rnn_optimizer, clip):
    rnn_optimizer.zero_grad()
    
    input_variables = input_variables.to(device)
    
    # create initial hidden_state
    hidden_state = (torch.zeros(1, 1, rnn.hidden_size).to(device), 
                    torch.zeros(1, 1, rnn.hidden_size).to(device))
    
    loss = 0.
    num_tokens = 0.
    pred_sent = []
    
    # forward batch of tokens through rnn one time step at a time
    for t in range(input_variables.size()[0]-1):
        output, hidden_state = rnn(input_variables[t], hidden_state)
        topi = output.argmax(dim=1)
        token_loss = loss_fun(rnn.embedding(topi), rnn.embedding(input_variables[t+1]))  
        loss += token_loss
        num_tokens += 1
        pred_sent.append(output.detach().numpy()) # shape=(seq_len, vocab_size)
    
    # perform backpropagation
    loss.backward()
    
    # clip gradients
    _ = torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip)
    
    # adjust model weights
    rnn_optimizer.step()
    
    return loss/num_tokens, pred_sent


def train_iters(iters, vocab_dict, batch_size, hidden_size, lr, norm_clipping):
    # load data
    sents_id = get_sent_id('trn-wiki.txt', vocab_dict)
    
    # build model
    rnn = SimpleRNN(vocab_size=len(vocab_dict.keys()), hidden_size=hidden_size)
    rnn.to(device)
    rnn.train()
    # initialize optimizer
    rnn_optimizer = torch.optim.SGD(rnn.parameters(), lr=lr)
    # initialize loss function
    loss_fun = nn.MSELoss()
    
    # start training
    for k in range(iters):
        avg_loss = 0.
        pred_sents = []
        for i in range(len(sents_id)):
            input_variables = get_sent_tensor(sents_id[i])
            loss, pred_sent = train(input_variables, loss_fun, rnn, rnn_optimizer, norm_clipping)
            pred_sents.append(pred_sent)    # shape=(total_num, seq_len, vocab_size)
            avg_loss += loss
        p = perplexity(pred_sents, sents_id)
        
        # Save checkpoint
        if i % 10 == 0:
            directory = 'checkpoints'
            if not os.path.exists(directory):
                os.makedirs(directory)
    
            torch.save({
                'iteration': i,
                'rnn': rnn.state_dict(),
                'rnn_opt': rnn_optimizer.state_dict(),
                'loss': loss,
                'embedding': rnn.embedding.state_dict(),
            }, os.path.join(directory, '{}_{}.tar'.format(k, 'checkpoint')))
    
        print("Iteration: {}; Average loss: {:.4f}; Perplexity: {}".format(k, avg_loss/len(sents_id), p))
    return rnn


def predict(rnn, vocab_dict):
    rnn.eval()
    
    # load data
    sents_id = get_sent_id('tst-wiki.txt', vocab_dict)
    outs = []
    new_dict = {v : k for k, v in vocab_dict.items()}
    for i in range(len(sents_id)):
        input_variables = get_sent_tensor(sents_id[i])
        # create initial hidden_state
        hidden_state = (torch.zeros(1, 1, rnn.hidden_size).to(device), 
                        torch.zeros(1, 1, rnn.hidden_size).to(device))
        pred_sent = ''
        # forward batch of tokens through rnn one time step at a time
        for t in range(input_variables.size()[0]-1):
            output, hidden_state = rnn(input_variables[t], hidden_state)
            prob = torch.log(torch.gather(output, 0, input_variables[t+1]))
            prob = prob.detach().numpy()[0]
            word_idx = input_variables[t+1].numpy()[0]
            pred_sent += new_dict[word_idx]+'\t'+str(prob)+' '
        outs.append(pred_sent)
    return outs


def validation(rnn, vocab_dict):
    rnn.eval()
    
    # load data
    sents_id = get_sent_id('tst-wiki.txt', vocab_dict)
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
            pred_sent.append(output.detach().numpy())    # shape=(seq_len, vocab_size)
        pred_sents.append(pred_sent)            # shape=(total_num, seq_len, vocab_size)
    p = perplexity(pred_sents, sents_id)
    print('Dev Perplexity:',p)


if __name__=='__main__':
    vocab_dict = {'<pad>':0, '<unk>':1, '<num>':2, '<start>':3, '<stop>':4}
    batch_size = 1
    hidden_size = 32
    lr = 0.001
    norm_clipping = 5.0
    
    trn_words, max_len = load_data('trn-wiki.txt')
    vocab_dict = build_vocab(trn_words, vocab_dict)
    
    rnn = train_iters(1, vocab_dict, batch_size, hidden_size, lr, norm_clipping)
    validation(rnn, vocab_dict)
    outs = predict(rnn, vocab_dict)
    with open('wd5jq-tst-logprob.txt', 'w', encoding='utf8') as f:
        for line in outs:
            f.write(line+'\n')