# 该模块主要实现的是对图像进行去噪的测试。
'''
# 首先第一步，引入相关的库函数
'''

import torch
from torch import nn
from config import *
from diffusion import alpha_t, alpha_bar
from dataset import *
import matplotlib.pyplot as plt
from diffusion import forward_diffusion
from lora import inject_lora
from lora import LoraLayer

'''
# 第二步定义一个去噪的函数
'''


def backward_denoise(net, batch_x_t, batch_labels):
    # 首先计算所需要的数据，方差variance,也就公式里面的beta_t
    alpha_bar_late = torch.cat((torch.tensor([1.0]), alpha_bar[:-1]), dim=0)
    variance = (1 - alpha_t) * (1 - alpha_bar_late) / (1 - alpha_bar)
    # 得到方差后，开始去噪
    net.eval()  # 开启测试模式
    # 记录每次得到的图像
    steps = [batch_x_t]
    for t in range(T - 1, -1, -1):
        # 初始化当前每张图像对应的时间状态
        batch_t = torch.full(size=(batch_x_t.size()[0],), fill_value=t)  # 表示此时的时间状态 (batch，)
        # 预测噪声
        # 修改第十四处
        batch_noise_pre = net(batch_x_t, batch_t, batch_labels)  # (batch,channel,iamg,imag)

        # 开始去噪（需要注意一个点，就是去噪的公式，在t不等于0和等于0是不一样的，先进行都需要处理部分也就是添加噪声前面的均值部分）
        # 同时记得要统一维度，便于广播
        reshape_size = (batch_t.size()[0], 1, 1, 1)
        # 先取出对应的数值
        alpha_t_batch = alpha_t[batch_t]
        alpha_bar_batch = alpha_bar[batch_t]
        variance_batch = variance[batch_t]
        # 计算前面的均值
        batch_mean_t = 1 / torch.sqrt(alpha_t_batch).reshape(*reshape_size) \
                       * (batch_x_t - (1 - alpha_t_batch.reshape(*reshape_size)) * batch_noise_pre / torch.sqrt(
            1 - alpha_bar_batch.reshape(*reshape_size)))

        # 分类，看t的值，判断是否添加噪声
        if t != 0:
            batch_x_t = batch_mean_t \
                        + torch.sqrt(variance_batch.reshape(*reshape_size)) \
                        * torch.randn_like(batch_x_t)
        else:
            batch_x_t = batch_mean_t

        # 对每次得到的结果进行上下限的限制
        batch_x_t = torch.clamp(batch_x_t, min=-1, max=1)
        # 添加每步的去噪结果
        steps.append(batch_x_t)
    return steps


# 开始测试
if __name__ == '__main__':
    # 加载模型
    model = torch.load('unet_epoch0.pt')
    model.eval()

    is_lora = False
    is_hebing = False
    # 如果是利用lora，需要把微调的也加进去模型进行推理
    if is_lora:
        for name, layer in model.named_modules():
            name_list = name.split('.')
            target_list = ['Wk', 'Wv', 'Wq']
            for i in target_list:
                if i in name_list and isinstance(layer, nn.Linear):
                    inject_lora(model, name, layer)

        # 加载权重参数
        try:
            para_load = torch.load('lora_para_epoch0.pt')
            model.load_state_dict(para_load, strict=False)
        except:
            pass

    # 如果需要合并，也就是把lora参数添加到原本的线性层上面的话，也就是把插入重新实现一遍，这次是把lora_layer换成linear。
    if is_lora and is_hebing:
        for name, layer in model:
            name_list = name.split('.')

            if isinstance(layer, LoraLayer):
                # 找到了对应的参数，把对应的lora参数添加到原本的参数上
                # 为什么要确定参数位置的上一层，因为setattr只能在上一层用，不能层层进入属性。
                cur_layer=model
                for n in name_list[:-1]:
                    cur_layer=getattr(cur_layer,n)
                # 首先计算lora参数
                lora_weight = torch.matmul(layer.lora_a, layer.lora_b) * layer.alpha / layer.r

                # 把参数进行添加,线性层的权重矩阵通常是 (out_features, in_features),所以需要对lora矩阵进行转置
                layer.net.weight = nn.Parameter(layer.net.weight.add(lora_weight.T))
                setattr(cur_layer, name_list[-1], layer)


    # 生成噪音图
    batch_size = 10
    batch_x_t = torch.randn(size=(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE))  # (5,1,48,48)
    batch_labels = torch.arange(start=0, end=10, dtype=torch.long)  # 引导词promot
    # 逐步去噪得到原图
    # 修改第十五处
    steps = backward_denoise(model, batch_x_t, batch_labels)
    # 绘制数量
    num_imgs = 20
    # 绘制还原过程
    plt.figure(figsize=(15, 15))
    for b in range(batch_size):
        for i in range(0, num_imgs):
            idx = int(T / num_imgs) * (i + 1)
            # 像素值还原到[0,1]
            final_img = (steps[idx][b] + 1) / 2
            # tensor转回PIL图
            final_img = TenosrtoPil_action(final_img)
            plt.subplot(batch_size, num_imgs, b * num_imgs + i + 1)
            plt.imshow(final_img)
    plt.show()
