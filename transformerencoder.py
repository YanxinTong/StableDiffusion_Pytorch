# 该模块实现的是交叉注意力机制，输入的是batch图像和batch标签的embdding,输出和batch_图像一样大小的融合了标签信息的图像。
# 修改的第一处。

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
from config import *

'''
# Part2 设置transformer编码器的类
'''


class TransformerEncoder(nn.Module):
    def __init__(self, img_channel, label_emd_size, q_k_size, v_size, f_size):
        super().__init__()
        # batch_x_t(batch,channel,imag_size,imag_size), label(batch,label_emd_size)
        # 我们这里是把图像的每个像素做为q(因此把channel移动到最后，转移为label_emd_size)，然后把label作为k和v的源头，转为label_emd_size

        # 初始化kqv矩阵
        self.Wq = nn.Linear(img_channel, q_k_size)
        self.Wk = nn.Linear(label_emd_size, q_k_size)
        self.Wv = nn.Linear(label_emd_size, v_size)

        # 此时得到了q (batch,imag_size*imag_size,q_k_size), k(batch,1,q_k_size), v(batch,1,v_size)
        # 记得对q,k矩阵归一化
        self.softmax1 = nn.Softmax(dim=-1)
        # 通过计算得到注意力(batch,imag_size * imag_size,v_size)
        # 因为要残差归一化，所以需要转化为原来形状
        # 转化为原本的大小
        self.linear1 = nn.Linear(v_size, img_channel)
        self.norm1 = nn.LayerNorm(img_channel)

        # 前向传播，feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(img_channel, f_size),
            nn.ReLU(),
            nn.Linear(f_size, img_channel)
        )
        # 残差归一化,得到结果
        self.norm2 = nn.LayerNorm(img_channel)

    def forward(self, batch_x_t, batch_labels_emd):
        batch_x_t = batch_x_t.permute(0, 2, 3, 1)  # (batch,imag_size,imag_size,channel)
        # 为了和交叉注意力机制保持统一，都需要满足是三维向量，其中第二维表示句子长度。

        batch_x_t_emd = batch_x_t.reshape(batch_x_t.size()[0], batch_x_t.size()[1] * batch_x_t.size()[2], batch_x_t.size()[3])
        q = self.Wq(batch_x_t_emd)  # (batch,imag_size* imag_size,k_q_size)
        k = self.Wk(batch_labels_emd).unsqueeze(1).transpose(1, 2)  # (batch,k_q_size,1)
        v = self.Wv(batch_labels_emd).unsqueeze(1)  # (batch,1,v_size)

        q_k = torch.matmul(q, k) / torch.sqrt(torch.tensor(q.size()[2])) # (batch,imag_size* imag_size,1)
        q_k_norm = self.softmax1(q_k)
        atten_z = self.linear1(torch.matmul(q_k_norm, v))  # (batch,imag_size*imag_size,img_channel)
        atten_z = atten_z.reshape(batch_x_t.size()[0],batch_x_t.size()[1],batch_x_t.size()[2],batch_x_t.size()[3])
        # 残差归一化
        atten_z = self.norm1(atten_z + batch_x_t)  # (batch,imag_size,imag_size,channel)

        # 前向以及残差归一化
        atten_z_feed = self.feedforward(atten_z)
        atten_z1 = self.norm2(atten_z_feed + atten_z)

        return atten_z1.permute(0, 3, 1, 2)


if __name__ == '__main__':
    batch_size = 2
    channel = 1
    qsize = 256
    cls_emb_size = 32

    cross_atn = TransformerEncoder(img_channel=1, q_k_size=256, v_size=128, f_size=512, label_emd_size=32)

    x = torch.randn((batch_size, channel, IMAGE_SIZE, IMAGE_SIZE))
    cls_emb = torch.randn((batch_size, cls_emb_size))  # cls_emb_size=32

    Z = cross_atn(x, cls_emb)
    print(Z.size())  # Z: (2,1,48,48)