import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import savemat
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from feature import Fea_Extra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
# torch.set_printoptions(threshold=np.inf)



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Declaring a variable "device" which will hold the device(i.e. either GPU or CPU) we are                                                                  #training the model on
# print(device)
device
# device = torch.device("cpu")
'''
num_epochs = 1 # Number of training epochs
d_model = 128  # dimension in encoder
heads = 4  # number of heads in multi-head attention
N = 2  # number of encoder layers
m = 14  # number of features
'''


class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.add_dimension=add_dimension()
        self.gating = CNN_Encoder()
        self.encoder = Encoder(d_model, N, heads,  dropout)
        # self.out = nn.Linear(d_model, 1)
        self.out = FC()

    def forward(self, src):
        src=self.add_dimension(src)
        e_i = self.gating(src)
        e_outputs = self.encoder(e_i)

        output = self.out(e_outputs)


        return output



class Encoder(nn.Module):
    def __init__(self, d_model, N, heads,
                 dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
        self.d_model = d_model
        self.flatten = nn.Flatten()

    def forward(self, src):
        # print(src)
        src = src.reshape(-1,1, self.d_model)  # this 128 is changed according to d_model
        x = src
        x = self.flatten(x)
        for i in range(self.N):
            x = self.layers[i](x, None)
        return self.norm(x)


# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])







# build an encoder layer with one multi-head attention layer and one # feed-forward layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.3):
        super().__init__()
        # self.norm_1 = Norm(d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        # self.norm_2 = Norm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = RCFeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1,keepdim=True)) / (
                    x.std(dim=-1,keepdim=True) + self.eps) + self.bias  # 减去均值除以方差

        return norm


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.3):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs,self.d_model)


        output = self.out(concat)

        return output


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
    # scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output




class RCFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        self.xiaorongconv1=conv_bn_relu(1, 2, 3, stride=1, padding=0,groups=1, bias=True, batch_norm=True)

        self.cancha=conv_bn_relu(1, 4, 3, stride=1,groups=1, batch_norm=True)

        self.flatten = nn.Flatten()



        self.xiaorongconv2=conv_bn_relu(2, 4, 3, stride=1,groups=1, padding=1, bias=True, batch_norm=True)

        #
        self.conv2 = nn.Conv1d(4, 2, 1, stride=1,padding=1)


    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1)

        x1=self.xiaorongconv1(x)

        x=self.cancha(x)


        x3 = self.xiaorongconv2(x1)


        x =x3+x
        x=self.conv2(x)

        x = self.flatten(x)

        x=self.dropout(x)
        x = self.linear_2(x)

        return x


def conv_bn_relu(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', batch_norm=True):
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     groups=groups, bias=bias, padding_mode=padding_mode)
    nn.init.xavier_uniform_(conv.weight)
    # leakrelu=nn.LeakyReLU()

    relu = nn.ReLU()
    # gelu=nn.GELU()
    if batch_norm:
        return nn.Sequential(

            conv,
            nn.BatchNorm1d(out_channels),
            relu
            # gelu
            # leakrelu
        )
    else:
        return nn.Sequential(
            conv,
            relu
            # gelu
            # leakrelu
        )







class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
        )

        self.predict = nn.Sequential(
            nn.Linear(in_features=64, out_features=1, bias=True),
        )


    def forward(self, x):


        x = self.hidden1(x)


        x=nn.ReLU()(x)

        x = self.predict(x)

        return x




class CNN_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool3 = nn.MaxPool1d(2, stride=2)

        self.conv11 = conv_bn_relu(1, 8, 3, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool11 = nn.AdaptiveAvgPool1d(1024)
        self.conv12 = conv_bn_relu(1, 8, 5, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool12 = nn.AdaptiveAvgPool1d(1024)
        self.conv13 = conv_bn_relu(1, 8, 7, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool13 = nn.AdaptiveAvgPool1d(1024)
        self.conv21 = conv_bn_relu(8, 16, 3, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool21 = nn.AdaptiveAvgPool1d(512)
        self.conv22 = conv_bn_relu(8, 16, 5, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool22 = nn.AdaptiveAvgPool1d(512)
        self.conv23 = conv_bn_relu(8, 16, 7, stride=1, padding=0, bias=True, batch_norm=True)
        self.avpool23 = nn.AdaptiveAvgPool1d(512)

        self.convendcan = conv_bn_relu(48, 2, 1, stride=4, padding=0, bias=True, batch_norm=True)
        self.convend1 = conv_bn_relu(48, 8, 3, stride=2, padding=2, bias=True, batch_norm=True)
        self.convend2 = conv_bn_relu(8, 2, 3, stride=2, padding=0, bias=True, batch_norm=True)
        self.maxpool=nn.MaxPool1d(2, stride=2)

        self.flatten = nn.Flatten()

        self.cbam=CBAMLayer(48,16,7)



        self.fc0 = nn.Linear(256, 128)#8128,2016,512

        self.dropout1 = nn.Dropout(p=0.3)

    def forward(self, x):


        x1 = self.conv11(x)  # [N*1*2560]
        x1=self.avpool11(x1)
        x1 = self.conv21(x1)
        x1 = self.avpool21(x1)


        x2 = self.conv12(x)
        x2 = self.avpool12(x2)
        x2 = self.conv22(x2)
        x2 = self.avpool22(x2)

        x3=self.conv13(x)
        x3 = self.avpool13(x3)
        x3 = self.conv23(x3)
        x3 = self.avpool23(x3)



        x = torch.cat((x1, x2,x3), 1)  # [N*256]
        x=self.cbam(x)

        xcan=self.convendcan(x)



        x=self.convend1(x)
        x=self.convend2(x)
        x=x+xcan
        x = self.flatten(x)


        x = self.fc0(x)  # [Nx128]
        x = nn.ReLU()(x)



        x = x.unsqueeze(-1)
        x=x.permute(0, 2, 1)

        return x


# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Declaring a variable "device" which will hold the device(i.e. either GPU or CPU) we are                                                                  #training the model on
# device
# model = Transformer(d_model=128,N=2,heads=4,dropout=0.1).to(device)
# print(model)

class add_dimension(nn.Module):
    def __init__(self):
        super(add_dimension, self).__init__()
        # self.fc=nn.Linear(32768,2560)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x=x.permute(0, 2, 1)

        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv1d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x










