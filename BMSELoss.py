#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-12
Modify Date: 2018-04-12
descirption: ""
'''
import torch


class BMSELoss(torch.nn.Module):

    def __init__(self):
        super(BMSELoss, self).__init__()
        self.w_l = [1, 2, 5, 10, 30]
        self.y_l = [0.283, 0.353, 0.424, 0.565, 1]

    def forward(self, x, y):
        w = y.clone()
        for i in range(len(self.w_l)):
            w[w < self.y_l[i]] = self.w_l[i]
        return torch.mean(w * ((y - x)** 2) )
