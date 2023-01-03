import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module): 
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,wv->ncwl',(x,A)) 
        return x.contiguous()


class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class gcn(nn.Module): 
    def __init__(self,c_in,c_out):
        """
        :param c_in: input channels
        :param c_out: output channels
        :param dropout:
        """
        super(gcn,self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out) 


    def forward(self,x,support): 
        # x=[64, 32, 207, 12]
        out = self.nconv(x,support) 
        # h = [64, 32, 207, 12]
        out = self.mlp(out)
        out = torch.sigmoid(out)
        # h=[64, 32, 207, 12]
        # out = F.dropout(out, self.dropout, training=self.training)
        return out
