import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module): # 做了一个4D张量与2D张量的乘积
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,wv->ncwl',(x,A)) # 爱因斯坦求和约定
        return x.contiguous()


class linear(nn.Module): # 线性层，这里做的其实是一个GCN中参数矩阵的学习
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)
        # 1*1卷积，补0策略，步长，有偏置（我觉得这个1*1卷积没必要补0呀）

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module): # 进行GCN
    def __init__(self,c_in,c_out):
        """
        :param c_in: 输入通道数
        :param c_out: 输出通道数
        :param dropout:
        """
        super(gcn,self).__init__()
        self.nconv = nconv()
        # 这个公式意思是我有几个邻接矩阵下的特征张量乘以n阶值得到n阶下的全部特征张量，再加上原先最原始的那个特征张量，再乘以每个特征张量下的通道数等于最终的通道数
        self.mlp = linear(c_in,c_out) # 线性层


    def forward(self,x,support): # 特征张量 邻接矩阵
        # x=[64, 32, 207, 12]

        out = self.nconv(x,support) # 特征矩阵乘以邻接矩阵
        # h = [64, 32, 207, 12]
        out = self.mlp(out) # 输入数据，这个应该是做的可学习的参数矩阵呀
        out = torch.sigmoid(out)
        # h=[64, 32, 207, 12]
        # out = F.dropout(out, self.dropout, training=self.training) # training=self.training必写才能使用
        return out
