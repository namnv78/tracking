'''
Pytorch implementation of SelecSLS Network architecture as described in
'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera, Mehta et al. 2019'.
The network architecture performs comparable to ResNet-50 while being 1.4-1.8x faster,
particularly with larger image sizes. The network architecture has a much smaller memory
footprint, and can be used as a drop in replacement for ResNet-50 in various tasks.
This Pytorch implementation establishes an official baseline of the model on ImageNet
This model also provides functionality to prune channels based on implicit sparsity, as
described in 'On Implicit Filter Level Sparsity in Convolutional Neural Networks, Mehta et al. CVPR 2019'.
This gives a 10-15% speedup depending on the model used.
Author: Dushyant Mehta (dmehta[at]mpi-inf.mpg.de)
This code is made available under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/legalcode)
'''
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import fractions
from .layers import Mish

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        Mish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        Mish()
    )

class SelecSLSBlock(nn.Module):
    def __init__(self, inp, skip, k, oup, isFirst, stride):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]

        #Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, k, 3, stride, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                Mish()
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(k, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                Mish()
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                Mish()
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(k//2, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                Mish()
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                Mish()
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(2*k + (0 if isFirst else skip), oup, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(oup),
                Mish()
                )

    def forward(self, x):
        assert isinstance(x,list)
        assert len(x) in [1,2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)) , x[1]]

class SelecSLS(nn.Module):
    def __init__(self, nClasses=1000, config='SelecSLS60'):
        super(SelecSLS, self).__init__()
	
        #Stem
        self.stem = conv_bn(3, 16, 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Core Network
        self.features = []
        if config=='SelecSLS42':
            print('SelecSLS42')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 144, 144,  True,  2],
                [144, 144, 144, 288,  False, 1],
                [288,   0, 304, 304,  True,  2],
                [304, 304, 304, 480,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(480, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS42_B':
            print('SelecSLS42_B')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 16,   0,  32,  32,  True,  2],
                [ 32,  32,  32, 48,  False, 1],
                [48,   0, 64, 64,  True,  2],
                [64, 64, 64, 128,  False, 1]
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(128, 480, 1),
#                     conv_bn(480, 512, 1),
#                     conv_bn(512, 640, 2),
                    conv_1x1_bn(480, 448),
                    )
            self.num_features = 448
        elif config=='SelecSLS60':
            print('SelecSLS60')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 416,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS60_B':
            print('SelecSLS60_B')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 416,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1280, 2),
                    conv_1x1_bn(1280, 1024),
                    )
            self.num_features = 1024
        elif config=='SelecSLS84':
            print('SelecSLS84')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 144,  False, 1],
                [144,   0, 144, 144,  True,  2],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 304,  False, 1],
                [304,   0, 304, 304,  True,  2],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 304,  False, 1],
                [304, 304, 304, 512,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(512, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        elif config=='SelecSLS102':
            print('SelecSLS102')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64,  64,  False, 1],
                [ 64,  64,  64,  64,  False, 1],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 480,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(480, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
            self.num_features = 1280
        else:
            raise ValueError('Invalid net configuration '+config+' !!!')

        #Build SelecSLS Core 
        for inp, skip, k, oup, isFirst, stride  in self.selecSLS_config:
            self.features.append(SelecSLSBlock(inp, skip, k, oup, isFirst, stride))
        self.features = nn.Sequential(*self.features)


    def extract_features(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.features([x])
        x = self.head(x[0])
        #x = x.mean(3).mean(2)
        #x = F.log_softmax(x)
        return x
    

def selecsls_60b(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SelecSLS(config='SelecSLS60_B')
    if pretrained:
        pretrained_dict = model_zoo.load_url('http://gvv.mpi-inf.mpg.de/projects/XNect/assets/models/SelecSLS60_B_statedict.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(pretrained_dict)
    
    return model

def selecsls_42b(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SelecSLS(config='SelecSLS42_B')
    if pretrained:
        pretrained_dict = torch.load('model/SelecSLS42_B_statedict.pth')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model.load_state_dict(pretrained_dict)
    
    return model
