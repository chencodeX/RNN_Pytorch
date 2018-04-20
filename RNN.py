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


def test(num_seqs, channels_img, size_image, max_epoch, model, cuda_test):
    input_image = torch.rand(num_seqs, 8, channels_img, size_image, size_image)
    input_gru = Variable(input_image.cuda())
    MSE_criterion = nn.MSELoss()
    for time in xrange(num_seqs):
        h_next = model(input_gru[time])


if __name__ == '__main__':
    num_seqs = 10
    hidden_size = 3
    channels_img = 2
    size_image = 120
    max_epoch = 10
    cuda_flag = False
    kernel_size = 3
    rcg = RNNCovnGRU(inplanes=2, input_num_seqs=10,output_num_seqs=10)
    print(rcg)
    rcg = rcg.cuda()
    test(num_seqs,channels_img,size_image,max_epoch,rcg,cuda_flag)



