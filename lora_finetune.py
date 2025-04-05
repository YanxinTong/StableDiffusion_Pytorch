# 该模块主要实现对于模型的一些模块进行微调训练，只对lora里面的新增参数进行训练。
'''
# Part 1 引入相关的库函数
'''
import os

import torch
from torch import nn
from dataset import minist_train
from torch.utils import data
from diffusion import forward_diffusion
from config import *
from unet import Unet
from lora import inject_lora


if __name__ =='__main__':
    '''
    # Part2 对需要训练的模型参数进行设置，将需要替换的线性层进行lora替换，并且只对lora进行训练
    '''
    # 首先第一步得先下载网络
    net = torch.load('unet_epoch0.pt')

    # 开始对所需的部分进行替换。
    # 首先，我们要对线性层进行lora替换，所以需要，输入inject_lora的参数包含(整个模型，路径，layer)
    for name, layer in net.named_modules():
        name_list = name.split('.')
        target = ['Wq', 'Wk', 'Wv']
        for i in target:
            if i in name_list and isinstance(layer, nn.Linear):
                # 替换
                inject_lora(net, name, layer)

    # 替换完之后，先看看需不需要添加之前的参数
    try:
        # 先下载参数
        lora_para=torch.load('lora_para_epoch0.pt')
        # 再填充到模型里面
        net.load_state_dict(lora_para,strict=False)

    except:
        pass


    # 替换完之后，需要对所有的参数进行设置，不是lora的参数梯度设置为False
    for name, para in net.named_parameters():
        name_list = name.split('.')
        lora_para_list = ['lora_a', 'lora_b']
        if name_list[-1] in lora_para_list:
            para.requires_grad = False
        else:
            para.requires_grad = True

    '''
    # Part3 进行训练
    '''
    epoch = 5
    batch_size = 50
    minist_loader = data.DataLoader(dataset=minist_train, batch_size=batch_size, shuffle=True)

    # 初始化模型
    loss = nn.L1Loss()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    n_iter = 0
    net.train()

    for i in range(epoch):
        for imgs, labels in minist_loader:
            imgs = imgs * 2 - 1
            # 先随机初始化batch_t
            batch_t = torch.randint(0, T, size=(imgs.size()[0],))
            # 首先对清晰图像进行加噪，得到batch_x_t
            batch_x_t, batch_noise = forward_diffusion(imgs, batch_t)

            # 预测对应的噪声
            batch_noise_pre = net(batch_x_t, batch_t, labels)

            # 计算损失
            l = loss(batch_noise, batch_noise_pre)

            # 清除梯度
            opt.zero_grad()
            # 损失反向传播
            l.backward()

            # 更新参数
            opt.step()

            # 累加损失
            last_loss = l.item()
            # 更新迭代次数
            n_iter += 1

            print('当前的iter为{},当前损失为{}'.format(n_iter, last_loss))
        print('当前的epoch为{},当前的损失为{}'.format(i, last_loss))

        # 保存训练好的lora参数，但是得先找到
        lora_dic = {}
        # 遍历net的参数
        for name, para in net.named_parameters():
            name_list = name.split('.')
            need_find = ['lora_a', 'lora_b']
            # 如果最后一个名字在需要找的参数里面
            if name_list[-1] in need_find:
                # 在存储的字典里面添加参数和名字
                lora_dic[name] = para
        # 先存储为临时文件
        torch.save(lora_dic, 'lora_para_epoch{}.pt.tmp'.format(i))
        # 然后改变路径，形成最终的参数(主要是为了防止写入出错)
        os.replace('lora_para_epoch{}.pt.tmp'.format(i), 'lora_para_epoch{}.pt'.format(i))
