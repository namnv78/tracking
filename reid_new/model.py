import torch
from torch import nn
from .selecsls import selecsls_42b
from torch.nn import Parameter
import math
import torch.nn.functional as F
from .layers import GeM
from torchsummary import summary



class Embedding(nn.Module):
    def __init__(self, in_channels, embedding_dimension):
        super().__init__()
        # Head
        self._pooling = GeM()
        self._bn1 = nn.BatchNorm1d(num_features=in_channels)
        self._dropout = nn.Dropout(0.5)
        self._fc = nn.Linear(in_channels, embedding_dimension)
        self._bn2 = nn.BatchNorm1d(num_features=embedding_dimension)
        self._bn2.bias.requires_grad_(False)
        #        self._pooling = GeM()
#         self._bn1 = nn.BatchNorm2d(num_features=in_channels)
#         self._dropout = nn.Dropout(0.5)
#         #self._fc = nn.Linear(in_channels, embedding_dimension)
#         self._fc = nn.Conv2d(in_channels, embedding_dimension, 1, 1, bias=False)
#         self._bn2 = nn.BatchNorm2d(num_features=embedding_dimension)
#         self._bn2.bias.requires_grad_(False)
        
        
    def forward(self, input):
        x = self._pooling(input)
        x = x.view(x.size(0), -1)
#         #print(x.size())
        x = self._bn1(x) 
        x = self._dropout(x)
        x = self._fc(x)
        x = self._bn2(x)
        #x = x.view(x.size(0), -1)
        x = F.normalize(x)
        return x


class ArcHead(nn.Module):
    def __init__(self, n_classes, in_channels, s=22, m=0.5):
        super().__init__()
        embedding_dimension = 128
        
        # Final linear layer
        self._embedding1 = Embedding(in_channels, embedding_dimension)
        
        # Global Logit
        self._embedding2 = Embedding(in_channels, embedding_dimension)
        self._dropout2 = nn.Dropout(0.5)
        if n_classes > 0:
            self._fc2 = nn.Linear(embedding_dimension, n_classes)

            # Arcface head
            self.weight = Parameter(torch.FloatTensor(n_classes, embedding_dimension))
            nn.init.orthogonal_(self.weight)

            self.s = s
            self.m = m
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            self.mm = math.sin(math.pi-m)*m
            self.threshold = math.cos(math.pi-m)

    def forward(self, input, label):
        embedding_norm1 = self._embedding1(input)*self.s

        weight_norm = F.normalize(self.weight)
        zy = F.linear(embedding_norm1, weight_norm).clamp(-self.s, self.s)
        cosine_t = zy/self.s
        sine_t = torch.sqrt((1.0 - cosine_t*cosine_t).clamp(0, 1))
        cos_phi = (cosine_t * self.cos_m - sine_t * self.sin_m)*self.s
        cos_phi = torch.where(cosine_t > self.threshold, cos_phi, zy - self.s * self.mm)
        
         # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine_t.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        arcface_output = (one_hot * cos_phi) + ((1.0 - one_hot) * zy)
        #output = one_hot*(cos_phi-zy) + zy
        
        embedding_norm2 = self._embedding2(input)
        classify_output = self._fc2(embedding_norm2)*self.s
        return arcface_output, classify_output


    def extract_features(self, input):
        embedding_norm1 = self._embedding1(input)
        return embedding_norm1



class Net(nn.Module):
    def __init__(self, n_classes, s=22, m=0.3):
        super(Net, self).__init__()
        self.conv_base = selecsls_42b(pretrained=False)
#         summary(self.conv_base, (1, 3,128,64))
        num_ftrs = 448
        #num_ftrs = self.conv_base._conv_head.out_channels
        # self.conv_base = resnet50(pretrained=True, filter_size=3)
        # num_ftrs = 2048
        #for param in self.conv_base.parameters():
        #    param.requires_grad = False
        self.head = ArcHead(n_classes, in_channels=num_ftrs, s=s, m=m)
        

    def forward(self, input, label):
        x = self.conv_base.extract_features(input)
#         summary(self.conv_base.extract_features, input.shape)
        arcface_output, classify_output = self.head(x, label)
        return arcface_output, classify_output

    def extract_features(self, input):
#     def forward(self, input):
#         input = input/255.0
#         input = input.transpose(1, 2).transpose(1, 3).contiguous()
        
        x = self.conv_base.extract_features(input)
        features = self.head.extract_features(x)
        return features
