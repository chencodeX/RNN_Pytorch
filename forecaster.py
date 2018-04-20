#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-30
Modify Date: 2018-03-30
descirption: ""
'''

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from ConvGRUCell import ConvGRUCell

def deconv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.ConvTranspose2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)

def conv2_act(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    layers = []
    layers += [nn.Conv2d(inplanes, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias)]
    layers += [nn.BatchNorm2d(out_channels)]
    layers += [nn.LeakyReLU(negative_slope=0.2)]
    return nn.Sequential(*layers)


def upsmaple(inplanes, out_channels=8, kernel_size=7, stride=5, padding=1, bias=True):
    # torch.cat((x, x, x), dim = 0)
    # the downsample layer input is last rnn output,like:output[-1]
    ret = deconv2_act(inplanes, out_channels, kernel_size, stride, padding, bias)
    return ret

class Forecaster(nn.Module):
    def __init__(self, num_seqs):
        super(Forecaster, self).__init__()

        num_filter = [64, 192, 192]
        kernel_size_l = [7, 7, 5]
        self.rnn1_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_1_h = None
        self.rnn1_2 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_2_h = None
        self.rnn1_3 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[2], kernel_size=kernel_size_l[2])
        self.rnn1_3_h = None
        #
        self.upsample1 = upsmaple(inplanes=num_filter[2], out_channels=num_filter[2], kernel_size=5, stride=3,
                                      padding=1)

        self.rnn2_1 = ConvGRUCell(input_size=num_filter[2], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_1_h = None
        self.rnn2_2 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_2_h = None
        self.rnn2_3 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[1], kernel_size=kernel_size_l[1])
        self.rnn2_3_h = None

        self.upsample2 = upsmaple(inplanes=num_filter[1], out_channels=num_filter[1], kernel_size=4, stride=2,
                                      padding=1 )

        self.rnn3_1 = ConvGRUCell(input_size=num_filter[1], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_1_h = None
        self.rnn3_2 = ConvGRUCell(input_size=num_filter[0], hidden_size=num_filter[0], kernel_size=kernel_size_l[0])
        self.rnn3_2_h = None

        self.deconv1 = deconv2_act(inplanes =num_filter[0] , out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.deconv2 = deconv2_act(inplanes=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv_final = conv2_act(inplanes=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.conv_pre = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)

    def set_h0(self,encoder):
        self.rnn1_1_h = encoder.rnn3_3_h
        self.rnn1_2_h = encoder.rnn3_2_h
        self.rnn1_3_h = encoder.rnn3_1_h
        self.rnn2_1_h = encoder.rnn2_3_h
        self.rnn2_2_h = encoder.rnn2_2_h
        self.rnn2_3_h = encoder.rnn2_1_h
        self.rnn3_1_h = encoder.rnn1_2_h
        self.rnn3_2_h = encoder.rnn1_1_h


    def forward(self,data):
        # print data.size()
        self.rnn1_1_h = self.rnn1_1(data,self.rnn1_1_h)
        self.rnn1_2_h = self.rnn1_1(self.rnn1_1_h, self.rnn1_2_h)
        self.rnn1_3_h = self.rnn1_1(self.rnn1_2_h, self.rnn1_3_h)
        # # print self.rnn1_3_h.size()
        data = self.upsample1(self.rnn1_3_h)
        # print data.size()
        self.rnn2_1_h = self.rnn2_1(data, self.rnn2_1_h)

        self.rnn2_2_h = self.rnn2_2(self.rnn2_1_h, self.rnn2_2_h)

        self.rnn2_3_h = self.rnn2_3(self.rnn2_2_h, self.rnn2_3_h)
        # print self.rnn2_3_h.size()
        data = self.upsample2(self.rnn2_3_h)
        # print data.size()
        self.rnn3_1_h = self.rnn3_1(data, self.rnn3_1_h)

        self.rnn3_2_h = self.rnn3_2(self.rnn3_1_h, self.rnn3_2_h)
        # print self.rnn3_2_h.size()
        data = self.deconv1(self.rnn3_2_h)
        # print data.size()
        # data = self.deconv2(data)
        # print data.size()
        data = self.conv_final(data)
        # print data.size()
        pre_data = self.conv_pre(data)
        # print pre_data.size()

        return pre_data


def test(num_seqs, channels_img, size_image, max_epoch, model, cuda_test):
    input_image = torch.rand(num_seqs, 8, channels_img, size_image, size_image)
    input_gru = Variable(input_image.cuda())
    MSE_criterion = nn.MSELoss()
    for time in xrange(num_seqs):
        h_next = model(input_gru[time])


if __name__ == '__main__':
    num_seqs = 10
    hidden_size = 3
    channels_img = 64
    size_image = 120
    max_epoch = 10
    cuda_flag = False
    kernel_size = 3
    enc = Forecaster(num_seqs=10)
    print(enc)
    enc =enc.cuda()
    test(num_seqs,channels_img,size_image,max_epoch,enc,cuda_flag)