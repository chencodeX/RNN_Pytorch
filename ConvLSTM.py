#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-20
Modify Date: 2018-04-20
descirption: ""
'''

import torch.nn as nn
from torch.autograd import Variable
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.dropout = nn.Dropout(p=0.5)
        # self.batch_size=batch_size
        self.padding = (filter_size - 1) / 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        # print type(hidden_state)
        hidden, c = hidden_state  # hidden and c are images with several channels
        # print 'hidden ',hidden.size()
        # print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)  # oncatenate in the channels
        # print 'combined',combined.size()
        # print type(combined.data)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        i = self.dropout(i)
        f = torch.sigmoid(af)
        f = self.dropout(f)
        o = torch.sigmoid(ao)
        o = self.dropout(o)
        g = torch.tanh(ag)
        g = self.dropout(g)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        next_h = self.dropout(next_h)
        return next_h, (next_h,next_c)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])).cuda(),
                Variable(torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1])).cuda())


class MultiConvRNNCell(nn.Module):
    def __init__(self,cells,state_is_tuple=True):
        super(MultiConvRNNCell, self).__init__()
        self._cells = cells
        self._state_is_tuple = state_is_tuple

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in xrange(len(self._cells)):
            init_states.append(self._cells[i].init_hidden(batch_size))
        return init_states

    def forward(self, input, hidden_state):
        cur_inp = input
        new_states = []
        for i, cell in enumerate(self._cells):
            cur_state = hidden_state[i]
            # print 'cur_inp size :', cur_inp.size()
            # print 'cur_state size :', cur_state[0].size()
            # print type(cur_inp.data),type(cur_state[0].data)
            cur_inp, new_state = cell(cur_inp, cur_state)
            # print 'cur_inp size :',cur_inp.size()
            # print 'cur_state size :', cur_state[0].size()
            new_states.append(new_state)

        new_states = tuple(new_states)
        return cur_inp,new_states



class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features, num_layers):
        super(CLSTM, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cuda())  # the first
        # one has a different number of input channels

        for idcell in xrange(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cuda())
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        # current_input = input.transpose(0, 1)  # now is seq_len,B,C,H,W
        current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in xrange(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            all_output = []
            output_inner = []
            for t in xrange(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            print output_inner[0].size()
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W
            print current_input.size()
        return next_hidden, current_input

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in xrange(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

if __name__ == '__main__':

    ###########Usage#######################################
    num_features = 64
    filter_size = 5
    batch_size = 8
    shape = (120, 120)  # H,W
    inp_chans = 3
    nlayers = 2
    seq_len = 10

    # If using this format, then we need to transpose in CLSTM
    input = Variable(torch.rand( seq_len,batch_size, inp_chans, shape[0], shape[1])).cuda()

    conv_lstm = CLSTM(shape, inp_chans, filter_size, num_features, nlayers)
    conv_lstm.apply(weights_init)
    conv_lstm.cuda()

    print 'convlstm module:', conv_lstm

    print 'params:'
    params = conv_lstm.parameters()
    for p in params:
        print 'param ', p.size()
        print 'mean ', torch.mean(p)

    hidden_state = conv_lstm.init_hidden(batch_size)
    print 'hidden_h shape ', len(hidden_state)
    print 'hidden_h shape ', hidden_state[0][0].size()
    out = conv_lstm(input, hidden_state)
    print 'out shape', out[1].size()
    print 'len hidden ', len(out[0])
    print 'next hidden', out[0][0][0].size()
    print 'convlstm dict', conv_lstm.state_dict().keys()

    # L = torch.sum(out[1])
    # L.backward()