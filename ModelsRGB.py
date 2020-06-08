
import resnetMod
from MyConvLSTMCell import *


class SupervisionHead(nn.Module):
    def __init__(self, in_channels, out_channels, h, w, m):
        super(SupervisionHead, self).__init__()
        self.conv = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, padding=0))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * out_channels, m * h * w))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class ConvLSTM(nn.Module):
    def __init__(self, supervision, num_classes=61, mem_size=512, loss_supervision="classification"):
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
        self.classifier = nn.Sequential(self.dropout, self.fc)
        if loss_supervision == "regression":
            self.sup_head = SupervisionHead(512, 100, 7, 7, 1)
        if loss_supervision == "classification":
            self.sup_head = SupervisionHead(512, 100, 7, 7, 2)

    def forward(self, x):
        state = (torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda())
        superv_x = []
        for t in range(x.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(x[t])
            state = self.lstm_cell(feature_convNBN, state)
            if self.supervision:
                superv_x.append(self.sup_head(feature_convNBN))
        if self.supervision:
            superv_x = torch.stack(superv_x)
            superv_x = superv_x.reshape(superv_x.shape[0] * superv_x.shape[1], -1, 7, 7)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, superv_x


class ConvLSTMAttention(nn.Module):
    def __init__(self, supervision, num_classes=61, mem_size=512, loss_supervision="classification"):
        super(ConvLSTMAttention, self).__init__()
        self.supervision = supervision
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        if loss_supervision == "regression":
            self.sup_head = SupervisionHead(512, 100, 7, 7, 1)
        if loss_supervision == "classification":
            self.sup_head = SupervisionHead(512, 100, 7, 7, 2)

    def forward(self, x):
        state = (torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda())
        superv_x = []
        for t in range(x.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(x[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h * w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attention_map = F.softmax(cam.squeeze(1), dim=1)
            attention_map = attention_map.view(attention_map.size(0), 1, 7, 7)
            attention_feat = feature_convNBN * attention_map.expand_as(feature_conv)
            state = self.lstm_cell(attention_feat, state)
            if self.supervision:
                superv_x.append(self.sup_head(feature_convNBN))
        if self.supervision:
            superv_x = torch.stack(superv_x)
            superv_x = superv_x.reshape(superv_x.shape[0] * superv_x.shape[1], -1, 7, 7)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, superv_x


class DynamicFilters(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DynamicFilters, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0))
        self.conv2 =nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        # batch,out_channel,in channel,size,size
        x_n = x.reshape(x.shape[0], self.out_channels // 3, 3, x.shape[2], x.shape[3])
        return x_n


class MyNet(nn.Module):
    def __init__(self, num_classes=61, mem_size=512):
        super(MyNet, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        self.dinam = DynamicFilters(512, 3 * 3)
        self.conv = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0),
                                  nn.ReLU())
        # self.convDynamic=   nn.Conv2d(3, 3, 3, stride=1,padding=2, bias=False)

    def forward(self, x):
        state = (torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda(),
                 torch.zeros((x.size(1), self.mem_size, 7, 7)).cuda())
        output_t = []
        for t in range(x.size(0)):
            output = []
            logit, feature_conv, feature_convNBN = self.resNet(x[t])
            
            
            
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h * w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
            attention_map = F.softmax(cam.squeeze(1), dim=1)
            attention_map = attention_map.view(attention_map.size(0), 1, 7, 7)
            attention_feat = feature_convNBN * attention_map.expand_as(feature_conv)
            
            
            state = self.lstm_cell(attention_feat, state)
            dynamic_filter = self.dinam(state[1])

            for i in range(x.shape[1]):
                output.append(F.conv2d((x[t][i].unsqueeze(0)),
                                       dynamic_filter[i].float(),
                                       padding=3))
            output_t.append(torch.stack(output))
        output_t = torch.stack(output_t)

        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        return feats, feats1, output_t.squeeze(2)
