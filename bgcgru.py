import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import gcn
from torch.autograd import Variable

class multi_range_attention(nn.Module):
    def __init__(self, c_out):
        super(multi_range_attention,self).__init__()
        self.w = nn.Parameter(torch.randn(c_out, c_out).cuda(), requires_grad=True).cuda()
        self.l = nn.Parameter(torch.randn(c_out, 1).cuda(), requires_grad=True).cuda()

    def forward(self,x):
        # x=b,f,n,t,m
        wx = torch.einsum('wf,bfntm->bwntm', (self.w, x))
        wxl = torch.einsum('bwntm,wv->bvntm', (wx,self.l))
        weight = F.softmax( wxl, dim=4) 

        x = torch.einsum('bvntm,bfntm->bfntm', (weight, x)) 
        x = torch.sum(x, dim=4) 
        return x # x=b,f,n,t


class mrabgcn(nn.Module):
    def __init__(self, c_in, c_out, layers=3):
        super(mrabgcn,self).__init__()

        self.gcn = nn.ModuleList()
        self.egcn = nn.ModuleList()
        self.layers = layers

        self.Wb = nn.Parameter(torch.randn(c_in, c_out).cuda(), requires_grad=True).cuda()

        for i in range(layers):
            if i == 0:
                self.gcn.append(gcn(c_in, c_out))
            else:
                self.gcn.append(gcn(c_out+c_out, c_out))
            if i == (self.layers - 1):
                break
            self.egcn.append(gcn(c_out, c_out))

        self.multi_range_attention = multi_range_attention(c_out)


    def forward(self, x, supports, M):
        input_g = x
        input_eg = torch.einsum('ne,bfnt->bfet', (M, x))
        input_eg = torch.einsum('bfet,fc->bcet', (input_eg, self.Wb))

        for i in range(self.layers):
            input_g = self.gcn[i](input_g,supports[0])

            try:
                add = torch.cat((add,input_g.unsqueeze(4)),dim=4)
            except:
                add = input_g.unsqueeze(4)

            if i == (self.layers-1):
                break

            input_eg = self.egcn[i](input_eg,supports[1])
            input_egtog = torch.einsum('ne,bfet->bfnt', (M, input_eg))
            input_g = torch.cat((input_g,input_egtog), dim=1)

        output = self.multi_range_attention(add)

        return output # b,f,n,t


class bgcgru(nn.Module):
    def __init__(self, c_in, c_out,supports, M, args):
        super(bgcgru, self).__init__()
        layers = args.layers
        self.gcn1 = mrabgcn(c_in+c_out, 2*c_out, layers)
        self.gcn2 = mrabgcn(c_in+c_out, c_out, layers)
        self.c_out = c_out
        self.c_in = c_in
        self.args = args
        self.supports = supports
        self.M = M

    def forward(self, x, hidden_state=None):
        """
        :inputs: shape (batch_size, num_nodes * input_dim)
        :output: shape (batch_size, num_nodes * output_dim)
        """
        hidden_state = hidden_state.view(self.args.batch_size, self.args.num_nodes, self.c_out) # (batch_size, num_nodes, c_out)
        hidden_state = hidden_state.permute(0, 2, 1)  # (batch_size, c_out, num_nodes)
        hidden_state = hidden_state.unsqueeze(3)  # (batch_size, c_out, num_nodes, 1)
        h = hidden_state

        x = x # (batch_size, num_nodes * input_dim)
        x = x.view(self.args.batch_size, self.args.num_nodes, self.c_in) # (batch_size, num_nodes,  input_dim)
        x = x.permute(0, 2, 1) # (batch_size, input_dim, num_nodes)
        x = x.unsqueeze(3) # (batch_size, input_dim, num_nodes, 1)

        input = x
        tem1 = torch.cat((input, h), 1)
        fea = self.gcn1(tem1, self.supports, self.M)
        z, r = torch.split(fea, [self.c_out, self.c_out], 1)
        tem2 = torch.cat((input, torch.sigmoid(r)*h), 1)
        c = self.gcn2(tem2, self.supports, self.M)
        new_h = torch.sigmoid(z) * h + (1-torch.sigmoid(z)) * torch.tanh(c)
        out = new_h

        out = out # (batch_size, fea, num_nodes, 1)
        out = out.permute(0, 2, 1, 3) # (batch_size, num_nodes, fea, 1)
        out = torch.reshape(out,(self.args.batch_size, self.args.num_nodes * self.c_out))
        # (batch_size, num_nodes * output_dim)

        return out # b,f,n,t

