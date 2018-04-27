#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-09
Modify Date: 2018-04-09
descirption: ""
'''
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import numpy as np
import sys
import cv2
import os
from ConvGRUCell import ConvGRUCell
from ConvLSTM import CLSTM_cell,MultiConvRNNCell


class RNNCovnGRU(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(RNNCovnGRU, self).__init__()
        self.inplanes =  inplanes
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        num_filter = 70
        kernel_size =7

        self.rnn1_1 = ConvGRUCell(input_size=2, hidden_size=num_filter, kernel_size=kernel_size)
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter, hidden_size=num_filter, kernel_size=kernel_size)
        self.rnn1_2_h = None
        # self.rnn1_3 = ConvGRUCell(input_size=num_filter, hidden_size=num_filter, kernel_size=kernel_size)
        # self.rnn1_3_h = None

        self.deconv1 = nn.ConvTranspose2d(num_filter, out_channels=1, kernel_size=3, stride=1, padding=1,
                         bias=True)

    def init_h0(self):
        self.rnn1_1_h = None
        self.rnn1_2_h = None
        # self.rnn1_3_h = None

    def forward(self,data):

        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data,self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_2(self.rnn1_1_h, self.rnn1_2_h)
        # self.rnn1_3_h = self.rnn1_3(self.rnn1_2_h, self.rnn1_3_h)
        # print self.rnn1_3_h.size()
        data = self.deconv1(self.rnn1_2_h)
        # print data.size()
        return data


class RNNConvLSTM(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs, shape):
        super(RNNConvLSTM, self).__init__()
        self.inplanes = inplanes
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.shape = (shape, shape)
        num_filter = 84
        kernel_size = 7

        self.cell1 = CLSTM_cell(self.shape, self.inplanes, kernel_size, num_filter)
        self.cell2 = CLSTM_cell(self.shape, num_filter, kernel_size, num_filter)


        self.stacked_lstm = MultiConvRNNCell([self.cell1,self.cell2])

        self.deconv1 = nn.ConvTranspose2d(num_filter, out_channels=1, kernel_size=3, stride=1, padding=1,
                                          bias=True)

    def forward(self, data):
        new_state = self.stacked_lstm.init_hidden(data.size()[1])
        # print new_state[0][0].size()
        # new_state = [(Variable(torch.zeros(8, 70, 120, 120).cuda()), Variable(torch.zeros(8, 70, 120, 120).cuda())),
        #              (Variable(torch.zeros(8, 70, 120, 120).cuda()), Variable(torch.zeros(8, 70, 120, 120).cuda()))]
        x_unwrap = []
        for i in xrange(self.input_num_seqs + self.output_num_seqs):
            # print i
            if i < self.input_num_seqs:
                y_1, new_state = self.stacked_lstm(data[i], new_state)
            else:
                y_1, new_state = self.stacked_lstm(x_1, new_state)
            # print y_1.size()
            x_1 = self.deconv1(y_1)
            # print x_1.size()
            if i >= self.input_num_seqs:
                x_unwrap.append(x_1)

        return x_unwrap


def test(num_seqs, channels_img, size_image, max_epoch, model, cuda_test):
    input_image = torch.rand(num_seqs, 4, channels_img, size_image, size_image)
    input_gru = Variable(input_image.cuda())
    MSE_criterion = nn.MSELoss()
    model = model.cuda()
    model.train()
    # new_state = model.stacked_lstm.init_hidden(8)
    # new_state = [(Variable(torch.zeros(8, 70, 120, 120).cuda()), Variable(torch.zeros(8, 70, 120, 120).cuda())),
    #              (Variable(torch.zeros(8, 70, 120, 120).cuda()), Variable(torch.zeros(8, 70, 120, 120).cuda()))]
    model(input_gru)
    # for time in xrange(num_seqs):
    #     h_next = model(input_gru[time])


if __name__ == '__main__':
    num_seqs = 10
    hidden_size = 3
    channels_img = 1
    size_image = 120
    max_epoch = 10
    cuda_flag = False
    kernel_size = 3
    rcg = RNNConvLSTM(inplanes=1, input_num_seqs=10, output_num_seqs=10, shape=size_image)
    print(rcg)
    # rcg = rcg.cuda()
    test(num_seqs, channels_img, size_image, max_epoch, rcg, cuda_flag)



