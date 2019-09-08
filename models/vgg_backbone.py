import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.box_utils import priorbox
from layers.inference import InferenceLayer
from layers.l2norm import L2NormScale


class SSD_VGG(nn.Module):
    def __init__(self, phase='train', num_classes=21):
        super(SSD_VGG, self).__init__()

        self.phase = phase
        self.num_classes = num_classes
        self.priors = Variable(priorbox('vgg'), volatile=True)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool_layer3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv_layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.l2_norm_scale = L2NormScale(512, 20)

        self.maxpool_layer4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv_layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.maxpool_layer5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv_layer14 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True))

        self.conv_layer15 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True))

        self.loc0 = nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1)
        self.class0 = nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1)

        self.loc1 = nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1)
        self.class1 = nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1)

        self.extra2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.loc2 = nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1)
        self.class2 = nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1)

        self.extra3 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True))
        self.loc3 = nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1)
        self.class3 = nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1)

        self.extra4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True))
        self.loc4 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.class4 = nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)

        self.extra5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True))
        self.loc5 = nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1)
        self.class5 = nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1)

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.inference_layer = InferenceLayer(top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, x):
        loc_list = []
        class_list = []

        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.maxpool_layer1(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.maxpool_layer2(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = self.conv_layer7(x)
        x = self.maxpool_layer3(x)
        x = self.conv_layer8(x)
        x = self.conv_layer9(x)
        x = self.conv_layer10(x)

        s = self.l2_norm_scale(x)
        loc_list.append(self.loc0(s))
        class_list.append(self.class0(s))

        x = self.maxpool_layer4(x)
        x = self.conv_layer11(x)
        x = self.conv_layer12(x)
        x = self.conv_layer13(x)
        x = self.maxpool_layer5(x)
        x = self.conv_layer14(x)
        x = self.conv_layer15(x)

        loc_list.append(self.loc1(x))
        class_list.append(self.class1(x))

        x = self.extra2(x)
        loc_list.append(self.loc2(x))
        class_list.append(self.class2(x))

        x = self.extra3(x)
        loc_list.append(self.loc3(x))
        class_list.append(self.class3(x))

        x = self.extra4(x)
        loc_list.append(self.loc4(x))
        class_list.append(self.class4(x))

        x = self.extra5(x)
        loc_list.append(self.loc5(x))
        class_list.append(self.class5(x))

        loc = []
        conf = []
        for (l, c) in zip(loc_list, class_list):
            loc.append(l.permute(0, 2, 3, 1).contiguous())
            conf.append(c.permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'train':
            output = (loc.view(loc.size(0), -1, 4),
                      conf.view(conf.size(0), -1, self.num_classes),
                      self.priors)
        else:
            conf = conf.view(conf.size(0), -1,
                             self.num_classes)
            for idx in range(conf.size(0)):
                conf[idx] = self.softmax(conf[idx])
            output = self.inference_layer(
                loc.view(loc.size(0), -1, 4),
                conf,
                self.priors.type(type(x.data)))

        return output
