# 该模块主要是实现lora类，实现lora层的alpha和beta通路，把输入的x经过两条通路后的结果，进行联合输出。
# 然后添加一个函数，主要是为了实现将原本的线性层换曾lora层。

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
from config import *

'''
# Part2 设计一个类，实现lora_layer
'''


class LoraLayer(nn.Module):
    def __init__(self, target_linear_layer, feature_in, feature_out, r, alpha):
        super().__init__()
        # 第一步，初始化lora的一些参数，包含a矩阵，b矩阵，r秩.比例系数等等。
        self.lora_a = nn.Parameter(torch.empty(feature_in, r), requires_grad=True)
        self.lora_b = nn.Parameter(torch.zeros(r, feature_out), requires_grad=True)
        self.alpha = alpha
        self.r = r

        # 第二步对alpha进行初始化
        nn.init.kaiming_uniform_(self.lora_a)

        # 第三步，初始化原本的目标线性层
        self.net = target_linear_layer

    def forward(self, x):
        output1 = self.net(x)
        output2 = torch.matmul(x, torch.matmul(self.lora_a, self.lora_b)) * (self.alpha / self.r)  # 得到结果后，乘上比例系数(alpha/r)
        return output2 + output1


'''
# Part3 定义一个函数，实现lora层的替换
'''


def inject_lora(module, name, target_linear_layer):  # 输入完整的模型，目标线性层的位置，目标线性层
    name_list = name.split('.')  # 按照.进行拆分路径
    # 获取到目标线性层的模型的上一层所有参数和模型{模型name1:模型，模型name2:模型}
    for i, item in enumerate(name_list[:-1]):
        module = getattr(module, item)
    # 初始化需要替换进入的lora层
    lora_layer = LoraLayer(target_linear_layer,
                           feature_in=target_linear_layer.in_features, feature_out=target_linear_layer.out_features,
                           r=LORA_R, alpha=LORA_ALPHA)
    # 替换对应的层
    setattr(module, name_list[-1], lora_layer)
