#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-02
Modify Date: 2018-04-02
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
from forecaster import Forecaster
from encoder import Encoder
from BMSELoss import BMSELoss
# sys.path.append('/home/meteo/zihao.chen/model_service/utils')
from data_transfrom import decode_radar_code, imgmap_tonumpy, encode_squs_code,imgmaps_tonumpy,imgmaps_tonumpy_crop

input_num_seqs = 10
output_num_seqs = 10
hidden_size = 3
input_channels_img = 1
output_channels_img = 1
size_image = 240
max_epoch = 4
cuda_flag = False
kernel_size = 3
batch_size = 2
lr= 0.0001
momentum = 0.5
# data = np.load('/home/meteo/zihao.chen/filter_ext/all_filter_data.npy')
# data = decode_radar_code(data)
# print data.shape
# code = 'AZ9010'
# train_arr, train_imgs_map, test_arr, test_imgs_map = encode_squs_code(data, code)
import pickle

# train_arr = np.load('train.npy')
# file_f = open('train.pkl', 'rb')
# train_imgs_map = pickle.load(file_f)
# file_f.close()
# test_arr = np.load('test.npy')
# file_f = open('test.pkl', 'rb')
# test_imgs_map = pickle.load(file_f)
# file_f.close()


class HKOModel(nn.Module):
    def __init__(self, inplanes, input_num_seqs, output_num_seqs):
        super(HKOModel, self).__init__()
        self.input_num_seqs = input_num_seqs
        self.output_num_seqs = output_num_seqs
        self.encoder = Encoder(inplanes=inplanes, num_seqs=input_num_seqs)
        self.forecaster = Forecaster(num_seqs=output_num_seqs)

    def forward(self, data):
        self.encoder.init_h0()
        for time in xrange(self.input_num_seqs):
            self.encoder(data[time])
        all_pre_data = []
        self.forecaster.set_h0(self.encoder)
        for time in xrange(self.output_num_seqs):
            pre_data = self.forecaster(None)
            # print h_next.size()
            all_pre_data.append(pre_data)

        return all_pre_data


def train_by_stype(model, loss, optimizer, x_val, y_val):
    # model.encoder.init_h0()
    # for time in xrange(model.input_num_seqs):
    #     h_next = model.encoder(x_val[time])
    #
    # all_pre_data = []
    #
    # model.forecaster.set_h0(model.encoder)
    #
    # for time in xrange(model.output_num_seqs):
    #     pre_data, h_next = model.forecaster(None)
    #     # print h_next.size()
    #     all_pre_data.append(pre_data)

    fx = model.forward(x_val)
    all_loss = 0

    for pre_id in range(len(fx)):
        output = loss.forward(fx[pre_id], y_val[pre_id])
        all_loss += output
        optimizer.zero_grad()
        output.backward(retain_graph=True)
        optimizer.step()
        # if pre_id == 1:
        #     print 'loss 1:',output
    return all_loss.cuda().data[0], fx


def train(model, loss, optimizer, x_val, y_val):
    # x = Variable(x_val.cuda(), requires_grad=False)
    # y = Variable(y_val.cuda(), requires_grad=False)
    optimizer.zero_grad()
    fx = model.forward(x_val)
    output = 0
    # t_y = fx.cpu().data.numpy().argmax(axis=1)
    # acc = 1. * np.mean(t_y == y_val.numpy())
    for pre_id in range(len(fx)):
        output += loss.forward(fx[pre_id], y_val[pre_id])
        # if pre_id == 1:
        #     print 'loss 1:',output
    output/=10.
    output.backward()
    optimizer.step()

    return output.cuda().data[0], fx


def verify(model, loss, x_val, y_val):
    fx = model.forward(x_val)
    output = 0
    for pre_id in range(len(fx)):
        output += loss.forward(fx[pre_id], y_val[pre_id])
    return output.cuda().data[0]


def load_data(code_list):
    test_arr = None
    train_arr = None
    train_imgs_maps = {}
    test_imgs_maps = {}
    for code in code_list:
        file_f = open('data_%s.pkl'%code,'rb')
        map_l = pickle.load(file_f)
        file_f.close()
        if test_arr is None:
            test_arr = map_l['test_arr']
            train_arr = map_l['train_arr']
        else:
            test_arr = np.concatenate((test_arr,map_l['test_arr']),axis=0)
            train_arr = np.concatenate((train_arr, map_l['train_arr']), axis=0)
        train_imgs_maps[code] = map_l['train_imgs_map']
        test_imgs_maps[code] = map_l['test_imgs_map']

    return train_arr,test_arr,train_imgs_maps,test_imgs_maps

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_t = lr
    lr_t = lr_t * (0.3 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_t


def touch_dir(path):
    result = False
    try:
        path = path.strip().rstrip("\\")
        if not os.path.exists(path):
            os.makedirs(path)
            result = True
        else:
            result = True
    except:
        result = False
    return result


def test(input_channels_img, output_channels_img, size_image, max_epoch, model, cuda_test):
    # input_image = np.ones((input_num_seqs,batch_size,input_channels_img,size_image,size_image))
    #
    # for i in range(input_num_seqs):
    #     input_image[i,...] = i*1+1
    # # input_image = input_image *10
    # input_image = torch.from_numpy(input_image).float()
    # input_gru = Variable(input_image.cuda())
    #
    # target_image = np.ones((output_num_seqs,batch_size,output_channels_img,size_image,size_image))
    # for i in range(output_num_seqs):
    #     target_image[i,...] = i*1+1*(input_num_seqs)+1
    #     print target_image[i,0,0,0:3,0:3]
    # # target_image = target_image *10
    # target_image = torch.from_numpy(target_image).float()
    # target_gru = Variable(target_image.cuda())
    # params = model.state_dict()
    # print params.keys()
    # print params['encoder.conv1_act.0.weight']
    criterion = nn.MSELoss()
    criterion = criterion.cuda()
    # optimizer = optim.SGD(model.parameters(), lr=(lr), momentum=0.9, weight_decay=0.005)
    optimizer = optim.Adam(model.parameters(), lr=(lr), weight_decay=0.005)
    print model
    print optimizer
    print criterion
    for i in range(max_epoch):
        # adjust_learning_rate(optimizer, i)
        print 'epoch :', i
        print train_arr.shape
        nnn = range(train_arr.shape[0])
        np.random.shuffle(nnn)
        train_arr_b = train_arr[nnn]
        batch_num = train_arr_b.shape[0] // batch_size
        print batch_num
        model.train()
        all_error = 0.
        for j in range(batch_num):
            batch_img = imgmaps_tonumpy_crop(train_arr_b[j * batch_size:(j + 1) * batch_size, ...], train_imgs_maps)
            input_image = batch_img[:10, ...] / 255.
            target_image = batch_img[10:,:,0,...] / 255.
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())

            error, pre_list = train(model, criterion, optimizer, input_gru, target_gru)
            all_error += error
            print j, ' : ', error
        print 'epoch train %d %f' % (i, all_error / batch_num)
        # print model.encoder.conv1_act
        # params = model.state_dict()
        # print params.keys()
        # print params['encoder.conv1_act.0.weight']
        batch_num = test_arr.shape[0] // batch_size
        model.eval()
        all_error = 0.
        for j in range(batch_num):
            batch_img = imgmaps_tonumpy(test_arr[j * batch_size:(j + 1) * batch_size, ...], test_imgs_maps)
            input_image = batch_img[:10, ...] / 255.
            target_image = batch_img[10:,:,0, ...] / 255.
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())

            error = verify(model, criterion, input_gru, target_gru)
            all_error += error
            print j, ' : ', error
        print 'epoch test %d %f' % (i, all_error / batch_num)
    model.eval()
    for i in range(test_arr.shape[0]):
        temp_path = test_arr[i, 0, 0]
        start_i = temp_path.find('201')
        time_str = temp_path[start_i:start_i + 12]
        print time_str
        start_i = temp_path.find('AZ')
        radar_code = temp_path[start_i:start_i + 6]
        save_path = '/home/meteo/zihao.chen/model_service/imgs/%s/%s/' % (radar_code,time_str)
        touch_dir(save_path)
        temp_arr = test_arr[i]
        temp_arr = temp_arr[np.newaxis, ...]
        batch_img = imgmaps_tonumpy(temp_arr, test_imgs_maps)
        input_image = batch_img[:10, ...]
        target_image = batch_img[10:,:,0, ...]
        input_image_t = torch.from_numpy(input_image / 255.).float()
        input_gru = Variable(input_image_t.cuda())
        fx = model.forward(input_gru)
        for pre_id in range(len(fx)):
            temp_xx = fx[pre_id].cpu().data.numpy()
            tmp_img = temp_xx[0, 0, ...]
            tmp_img = tmp_img * 255.
            true_img = target_image[pre_id, 0,  ...]
            encode_img = input_image[pre_id, 0, 0, ...]
            cv2.imwrite(os.path.join(save_path, 'a_%s.png' % pre_id), encode_img)
            cv2.imwrite(os.path.join(save_path, 'c_%s.png' % pre_id), tmp_img)
            cv2.imwrite(os.path.join(save_path, 'b_%s.png' % pre_id), true_img)

    # for pre_data in pre_list:
    #     temp = pre_data.cpu().data.numpy()
    #     print temp.mean()
train_arr,test_arr,train_imgs_maps,test_imgs_maps = load_data(['AZ9010','AZ9200'])

if __name__ == '__main__':
    m = HKOModel(inplanes=2, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    m = m.cuda()

    test(input_channels_img, output_channels_img, size_image, max_epoch, model=m, cuda_test=cuda_flag)
