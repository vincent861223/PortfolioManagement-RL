
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, rnn_hidden_size=128, n_layer=2, rnn_dropout=0.1):
        super(Actor, self).__init__()
        n_stocks = nb_states[1]
        n_time = nb_states[2]
        n_action = nb_actions[0]
        self.fc1 = nn.Linear(n_stocks * n_time, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_action)
        self.rnn = nn.GRU(n_stocks, rnn_hidden_size, n_layer, batch_first=True, dropout=rnn_dropout)
        self.fc = nn.Sequential(
                                nn.Linear(rnn_hidden_size, rnn_hidden_size//2),
                                nn.ReLU(),
                                nn.Linear(rnn_hidden_size//2, n_action)
                                )
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        # x: [batch, ohcl, n_stocks, n_time]

        #print(x)
        #print(x.size())
        # x = x[:, -2].view(x.size(0), -1);
        # #print(x.size())
        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.fc3(out)
        # out = self.tanh(out)
        # out = self.softmax(out)

        x = x[:, -2].permute(0, 2, 1)
        out, rnn_hidden = self.rnn(x)
        rnn_hidden = rnn_hidden.permute(1, 0, 2) #[batch, num_layers * num_directions, hidden_size]
        logit = self.fc(rnn_hidden.mean(dim=1))
        action = self.softmax(logit)

        return action

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3, rnn_hidden_size=128, n_layer=2, rnn_dropout=0.1):
        super(Critic, self).__init__()
        n_stocks = nb_states[1]
        n_time = nb_states[2]
        n_action = nb_actions[0]
        self.fc1 = nn.Linear(n_stocks * n_time, hidden1)
        self.fc2 = nn.Linear(hidden1+n_action, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.rnn = nn.GRU(n_stocks, rnn_hidden_size, n_layer, batch_first=True, dropout=rnn_dropout)
        self.fc = nn.Sequential(
                                nn.Linear(rnn_hidden_size + n_action, rnn_hidden_size // 2),
                                nn.ReLU(),
                                nn.Linear(rnn_hidden_size // 2, 1)
                                )
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        # #print(a.size())
        # x = x[:, -2].view(x.size(0), -1);
        # out = self.fc1(x)
        # out = self.relu(out)
        # # debug()
        # out = self.fc2(torch.cat([out,a],1))
        # out = self.relu(out)
        # out = self.fc3(out)

        x = x[:, -2].permute(0, 2, 1)
        out, rnn_hidden = self.rnn(x)
        rnn_hidden = rnn_hidden.permute(1, 0, 2) #[batch, num_layers * num_directions, hidden_size]
        out = self.fc(torch.cat([rnn_hidden.mean(dim=1), a], dim=1))

        return out