#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-04-08
Modify Date: 2018-04-08
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
import pickle
from forecaster import Forecaster
from encoder import Encoder
# sys.path.append('/home/meteo/zihao.chen/model_service/utils')
from data_transfrom import decode_radar_code, imgmap_tonumpy, encode_squs_code, imgmaps_tonumpy

input_num_seqs = 10
output_num_seqs = 10
hidden_size = 3
input_channels_img = 1
output_channels_img = 1
size_image = 240
max_epoch = 12
cuda_flag = False
kernel_size = 3
batch_size = 16


def train_by_stype(model_e, model_f, loss_e, loss_f, optimizer_e, optimizer_f, x_val, y_val):
    model_e.init_h0()
    for time in xrange(model_e.num_seqs):
        h_next_e = model_e(x_val[time])

    all_pre_data = []
    # print type(model_e)
    # print type(model_f)
    model_f.set_h0(model_e)

    for time in xrange(model_e.num_seqs):
        pre_data, h_next = model_f(None)
        # print h_next.size()
        all_pre_data.append(pre_data)

    # fx = model_f.forward(x_val)
    output_f = 0
    # print all_pre_data[0].requires_grad
    # print y_val[0].requires_grad
    optimizer_f.zero_grad()
    for pre_id in range(len(all_pre_data)):
        # print all_pre_data[pre_id].dtype()
        output_f += loss_f.forward(all_pre_data[pre_id], y_val[pre_id])

    output_f.backward(retain_graph=True)
    optimizer_f.step()
    all_e_rnn_h = [model_e.rnn3_3_h, model_e.rnn3_2_h, model_e.rnn3_1_h, model_e.rnn2_3_h, model_e.rnn2_2_h,
                   model_e.rnn2_1_h, model_e.rnn1_2_h, model_e.rnn1_1_h]

    all_f_rnn_h = [model_f.rnn1_1_h, model_f.rnn1_2_h, model_f.rnn1_3_h, model_f.rnn2_1_h, model_f.rnn2_2_h,
                   model_f.rnn2_3_h, model_f.rnn3_1_h, model_f.rnn3_2_h]
    output_e = 0
    optimizer_e.zero_grad()
    for i in range(len(all_e_rnn_h)):

    # all_f_rnn_h[0].requires_grad = False
        output_e += loss_e.forward(all_e_rnn_h[i],all_f_rnn_h[i].detach())
    output_e.backward()
    optimizer_e.step()

    # if pre_id == 1:
    #     print 'loss 1:',output
    return output_f.cuda().data[0], all_pre_data


# def train(model_e,model_f, loss, optimizer, x_val, y_val):
#     # x = Variable(x_val.cuda(), requires_grad=False)
#     # y = Variable(y_val.cuda(), requires_grad=False)
#     optimizer.zero_grad()
#     fx = model.forward(x_val)
#     output = 0
#     # t_y = fx.cpu().data.numpy().argmax(axis=1)
#     # acc = 1. * np.mean(t_y == y_val.numpy())
#     for pre_id in range(len(fx)):
#         output += loss.forward(fx[pre_id], y_val[pre_id]).data.cpu()
#         # if pre_id == 1:
#         #     print 'loss 1:',output
#     output.backward()
#     optimizer.step()
#
#     return output.cuda().data[0], fx


def verify(model_e, model_f, loss, x_val, y_val):
    model_e.init_h0()
    for time in xrange(model_e.num_seqs):
        h_next = model_e(x_val[time])

        fx = []

    model_f.set_h0(model_e)

    for time in xrange(model_e.num_seqs):
        pre_data, h_next = model_f(None)
        # print h_next.size()
        fx.append(pre_data)

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
        file_f = open('data_%s.pkl' % code, 'rb')
        map_l = pickle.load(file_f)
        file_f.close()
        if test_arr is None:
            test_arr = map_l['test_arr']
            train_arr = map_l['train_arr']
        else:
            test_arr = np.concatenate((test_arr, map_l['test_arr']), axis=0)
            train_arr = np.concatenate((train_arr, map_l['train_arr']), axis=0)
        train_imgs_maps[code] = map_l['train_imgs_map']
        test_imgs_maps[code] = map_l['test_imgs_map']

    return train_arr, test_arr, train_imgs_maps, test_imgs_maps


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.001
    lr = lr * (0.3 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


def test(input_channels_img, output_channels_img, size_image, max_epoch, model_e, model_f, cuda_test):
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
    params = model_e.state_dict()
    print params.keys()
    print params['conv1_act.0.weight']
    criterion_e = nn.MSELoss()
    criterion_e = criterion_e.cuda()
    optimizer_e = optim.SGD(model_e.parameters(), lr=(0.001), momentum=0.9, weight_decay=0.005)

    criterion_f = nn.MSELoss()
    criterion_f = criterion_f.cuda()
    optimizer_f = optim.SGD(model_f.parameters(), lr=(0.001), momentum=0.9, weight_decay=0.005)

    for i in range(max_epoch):
        adjust_learning_rate(optimizer_e, i)
        adjust_learning_rate(optimizer_f, i)
        print 'epoch :', i
        print train_arr.shape
        nnn = range(train_arr.shape[0])
        np.random.shuffle(nnn)
        train_arr_b = train_arr[nnn]
        batch_num = train_arr_b.shape[0] // batch_size
        print batch_num
        for j in range(batch_num):
            batch_img = imgmaps_tonumpy(train_arr_b[j * batch_size:(j + 1) * batch_size, ...], train_imgs_maps)
            input_image = batch_img[:10, ...] / 255.
            target_image = batch_img[10:, ...] / 255.
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())

            error, pre_list = train_by_stype(model_e, model_f, criterion_e, criterion_f, optimizer_e, optimizer_f,
                                             input_gru, target_gru)
            print j, ' : ', error
        # print model.encoder.conv1_act
        params = model_e.state_dict()
        print params.keys()
        print params['conv1_act.0.weight']
        batch_num = test_arr.shape[0] // batch_size
        for j in range(batch_num):
            batch_img = imgmaps_tonumpy(test_arr[j * batch_size:(j + 1) * batch_size, ...], test_imgs_maps)
            input_image = batch_img[:10, ...] / 255.
            target_image = batch_img[10:, ...] / 255.
            input_image = torch.from_numpy(input_image).float()
            input_gru = Variable(input_image.cuda())
            target_image = torch.from_numpy(target_image).float()
            target_gru = Variable(target_image.cuda())

            error = verify(model_e,model_f, criterion_f, input_gru, target_gru)
            print j, ' : ', error

    for i in range(test_arr.shape[0]):
        temp_path = test_arr[i, 0, 0]
        start_i = temp_path.find('201')
        time_str = temp_path[start_i:start_i + 12]
        print time_str
        start_i = temp_path.find('AZ')
        radar_code = temp_path[start_i:start_i + 6]
        save_path = '/home/meteo/zihao.chen/model_service/imgs/%s/%s/' % (radar_code, time_str)
        touch_dir(save_path)
        temp_arr = test_arr[i]
        temp_arr = temp_arr[np.newaxis, ...]
        batch_img = imgmaps_tonumpy(temp_arr, test_imgs_maps)
        input_image = batch_img[:10, ...]
        target_image = batch_img[10:, ...]
        input_image_t = torch.from_numpy(input_image / 255.).float()
        input_gru = Variable(input_image_t.cuda())
        # fx = model.forward(input_gru)
        model_e.init_h0()
        for time in xrange(model_e.num_seqs):
            h_next = model_e(input_gru[time])

        fx = []

        model_f.set_h0(model_e)

        for time in xrange(model_e.num_seqs):
            pre_data, h_next = model_f(None)
            # print h_next.size()
            fx.append(pre_data)

        for pre_id in range(len(fx)):
            temp_xx = fx[pre_id].cpu().data.numpy()
            tmp_img = temp_xx[0, 0, ...]
            tmp_img = tmp_img * 255.
            true_img = target_image[pre_id, 0, 0, ...]
            encode_img = input_image[pre_id, 0, 0, ...]
            cv2.imwrite(os.path.join(save_path, 'a_%s.png' % pre_id), encode_img)
            cv2.imwrite(os.path.join(save_path, 'c_%s.png' % pre_id), tmp_img)
            cv2.imwrite(os.path.join(save_path, 'b_%s.png' % pre_id), true_img)

    # for pre_data in pre_list:
    #     temp = pre_data.cpu().data.numpy()
    #     print temp.mean()


train_arr, test_arr, train_imgs_maps, test_imgs_maps = load_data(['AZ9010','AZ9200'])

if __name__ == '__main__':
    # m = HKOModel(inplanes=1, input_num_seqs=input_num_seqs, output_num_seqs=output_num_seqs)
    m_e = Encoder(inplanes=input_channels_img, num_seqs=input_num_seqs)
    m_e = m_e.cuda()

    m_f = Forecaster(num_seqs=output_num_seqs)
    m_f = m_f.cuda()

    test(input_channels_img, output_channels_img, size_image, max_epoch, model_e=m_e, model_f=m_f, cuda_test=cuda_flag)
