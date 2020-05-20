

import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *

class SupervisionHead(nn.Module):
    def __init__(self,in_channels,out_channels,h,w):
        super(SupervisionHead, self).__init__()
        self.conv= nn.Sequential(nn.ReLU(),
        		nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0))
        self.fc= nn.Sequential(nn.Linear(h*w*out_channels,2*h*w))
    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x

class ConvLSTM(nn.Module):
    def __init__(self, supervision,num_classes=61, mem_size=512):
        super(ConvLSTM, self).__init__()
        self.num_classes = num_classes
        self.supervision = supervision
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout,self.fc)
        self.sup_head= SupervisionHead(512,100,7,7)

    def forward(self, inputVariable,supervision):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        superv_x=[]
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            state = self.lstm_cell(feature_convNBN, state)
            if self.supervision==True:
                superv_x.append(self.sup_head(feature_convNBN))
        if self.supervision==True:
            superv_x=torch.stack(superv_x)
            superv_x= superv_x.reshape(superv_x.shape[0]*superv_x.shape[1],2,7,7)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, superv_x

class ConvLSTMAttention(nn.Module):
    def __init__(self, supervision,num_classes=61, mem_size=512):
        super(ConvLSTMAttention, self).__init__()
        self.supervision=supervision
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout,self.fc)
        self.sup_head= SupervisionHead(512,100,7,7)

    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        superv_x=[]
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h*w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attentionMAP = F.softmax(cam.squeeze(1), dim=1)
            attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
            attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
            state = self.lstm_cell(attentionFeat, state)
            if self.supervision==True:
                superv_x.append(self.sup_head(feature_convNBN))
        if self.supervision==True:
            superv_x=torch.stack(superv_x)
            superv_x= superv_x.reshape(superv_x.shape[0]*superv_x.shape[1],2,7,7)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, superv_x
        


class SupervisedLSTM(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(selfSuperAttentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout,self.fc)
        self.sup_head= SupervisionHead(512,100,7,7)

    def forward(self, inputVariable,supervision=False):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        superv_x=[]
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            state = self.lstm_cell(feature_conv, state)
            if supervision==True:
                superv_x.append(self.sup_head(state[0]))#len_seq,batch,img
        if supervision==True:
            superv_x=torch.stack(superv_x)
            superv_x= superv_x.reshape(superv_x.shape[0],superv_x.shape[1],7,7)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, superv_x



