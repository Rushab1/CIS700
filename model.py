import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from batch import *
from IPython import embed

class IntentModel(nn.Module):
    def __init__(self, dim=1024, bert_dim = 768, dropout=0.5):
        super(IntentModel, self).__init__()
        self.dim = dim
        self.bert_dim = bert_dim
        self.dropout = dropout

        self.linear = nn.Sequential()
        self.linear.add_module("drop1", nn.Dropout(self.dropout))
        self.linear.add_module("linear", nn.Linear(dim, bert_dim))
        self.linear.add_module("activation", nn.ReLU())

    def forward(self, x, o, t ):
        batchSize = len(t)
        output1 = self.linear(x).unsqueeze(1).transpose(1,2)
        output2 = torch.bmm(o, output1)
        output2 = output2.squeeze(2)
        return output2

class Optim:
    epoch_loss = 0
    def __init__(self, model, criterion = nn.CrossEntropyLoss(), lr = 0.0001, weight_decay=0):
        self.model = model
        self.criterion = criterion;
        self.optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay

    def update_lr(self, lr, weight_decay = None):
        if weight_decay == None:
            weight_decay = self.weight_decay
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.lr = lr

    def backward(self, output, y):
        loss = self.criterion(output, y)
        self.loss = loss
        #print(loss.item())
        self.epoch_loss += loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
