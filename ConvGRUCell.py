#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: ""
'''
import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


class ConvGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=0.5)
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size, 2 * self.hidden_size, self.kernel_size,
                                   padding=self.kernel_size // 2)
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.kernel_size // 2)
        dtype = torch.FloatTensor

    def forward(self, input, hidden):
        if hidden is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            # print size_h
            hidden = Variable(torch.zeros(size_h).cuda())
        if input is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            # print size_h
            input = Variable(torch.zeros(size_h).cuda())
        # print input.size()
        # print hidden.size()
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        (rt, ut) = c1.chunk(2, 1)
        reset_gate = self.dropout(f.sigmoid(rt))
        update_gate = self.dropout(f.sigmoid(ut))
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        ct = f.tanh(p1)
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        return next_h


