import torch
import torch.nn as nn
from utils.box_utils import priorbox
from layers.inference import InferenceLayer
from torch.autograd import Variable


class SSD_MobileNetV2(nn.Module):
    def __init__(self, phase='train', num_classes=2):
        super(SSD_MobileNetV2, self).__init__()

        self.phase = phase
        self.num_classes = num_classes
        self.priors = Variable(priorbox('mobilenetv2'), volatile=True)

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True))

        self.invertedresidual_layer1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, 1, 0, bias=False),
            nn.BatchNorm2d(16))

        self.invertedresidual_layer2 = nn.Sequential(
            nn.Conv2d(16, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, 2, 1, groups=96, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24))

        self.invertedresidual_layer3 = nn.Sequential(
            nn.Conv2d(24, 144, 1, 1, 0, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 144, 3, 1, 1, groups=144, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 24, 1, 1, 0, bias=False),
            nn.BatchNorm2d(24))

        self.invertedresidual_layer4 = nn.Sequential(
            nn.Conv2d(24, 144, 1, 1, 0, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 144, 3, 2, 1, groups=144, bias=False),
            nn.BatchNorm2d(144),
            nn.ReLU(inplace=True),
            nn.Conv2d(144, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32))

        self.invertedresidual_layer5 = nn.Sequential(
            nn.Conv2d(32, 192, 1, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32))

        self.invertedresidual_layer6 = nn.Sequential(
            nn.Conv2d(32, 192, 1, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 1, 1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 32, 1, 1, 0, bias=False),
            nn.BatchNorm2d(32))

        self.invertedresidual_layer7 = nn.Sequential(
            nn.Conv2d(32, 192, 1, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, 2, 1, groups=192, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64))

        self.invertedresidual_layer8 = nn.Sequential(
            nn.Conv2d(64, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64))

        self.invertedresidual_layer9 = nn.Sequential(
            nn.Conv2d(64, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64))

        self.invertedresidual_layer10 = nn.Sequential(
            nn.Conv2d(64, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, 1, 1, 0, bias=False),
            nn.BatchNorm2d(64))

        self.invertedresidual_layer11 = nn.Sequential(
            nn.Conv2d(64, 384, 1, 1, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, 1, 1, groups=384, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96))

        self.invertedresidual_layer12 = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 576, 3, 1, 1, groups=576, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96))

        self.invertedresidual_layer13 = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 576, 3, 1, 1, groups=576, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 96, 1, 1, 0, bias=False),
            nn.BatchNorm2d(96))

        self.regression_layer1 = nn.Sequential(nn.Conv2d(96, 96,
                                                         kernel_size=3,
                                                         groups=96, stride=1, padding=1),
                                               nn.BatchNorm2d(96),
                                               nn.ReLU(),
                                               nn.Conv2d(96, 6 * 4,
                                                         kernel_size=1))

        self.confidence_layer1 = nn.Sequential(nn.Conv2d(96, 96,
                                                         kernel_size=3,
                                                         groups=96, stride=1, padding=1),
                                               nn.BatchNorm2d(96),
                                               nn.ReLU(),
                                               nn.Conv2d(96, 6 * self.num_classes,
                                                         kernel_size=1))

        self.invertedresidual_layer14 = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 576, 3, 2, 1, groups=576, bias=False),
            nn.BatchNorm2d(576),
            nn.ReLU(inplace=True),
            nn.Conv2d(576, 160, 1, 1, 0, bias=False),
            nn.BatchNorm2d(160))

        self.invertedresidual_layer15 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 160, 1, 1, 0, bias=False),
            nn.BatchNorm2d(160))

        self.invertedresidual_layer16 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 160, 1, 1, 0, bias=False),
            nn.BatchNorm2d(160))

        self.invertedresidual_layer17 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 960, 3, 1, 1, groups=960, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True),
            nn.Conv2d(960, 320, 1, 1, 0, bias=False),
            nn.BatchNorm2d(320))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True))

        self.regression_layer2 = nn.Sequential(nn.Conv2d(1280, 1280,
                                                         kernel_size=3,
                                                         groups=1280, stride=1, padding=1),
                                               nn.BatchNorm2d(1280),
                                               nn.ReLU(),
                                               nn.Conv2d(1280, 6 * 4,
                                                         kernel_size=1))

        self.confidence_layer2 = nn.Sequential(nn.Conv2d(1280, 1280,
                                                         kernel_size=3,
                                                         groups=1280, stride=1, padding=1),
                                               nn.BatchNorm2d(1280),
                                               nn.ReLU(),
                                               nn.Conv2d(1280, 6 * self.num_classes,
                                                         kernel_size=1))

        self.extra1 = nn.Sequential(nn.Conv2d(1280, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(512))

        self.regression_layer3 = nn.Sequential(nn.Conv2d(512, 512,
                                                         kernel_size=3,
                                                         groups=512, stride=1, padding=1),
                                               nn.BatchNorm2d(512),
                                               nn.ReLU(),
                                               nn.Conv2d(512, 6 * 4,
                                                         kernel_size=1))

        self.confidence_layer3 = nn.Sequential(nn.Conv2d(512, 512,
                                                         kernel_size=3,
                                                         groups=512, stride=1, padding=1),
                                               nn.BatchNorm2d(512),
                                               nn.ReLU(),
                                               nn.Conv2d(512, 6 * self.num_classes,
                                                         kernel_size=1))

        self.extra2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256))

        self.regression_layer4 = nn.Sequential(nn.Conv2d(256, 256,
                                                         kernel_size=3,
                                                         groups=256, stride=1, padding=1),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Conv2d(256, 4 * 4,
                                                         kernel_size=1))

        self.confidence_layer4 = nn.Sequential(nn.Conv2d(256, 256,
                                                         kernel_size=3,
                                                         groups=256, stride=1, padding=1),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Conv2d(256, 4 * self.num_classes,
                                                         kernel_size=1))

        self.extra3 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256))

        self.regression_layer5 = nn.Sequential(nn.Conv2d(256, 256,
                                                         kernel_size=3,
                                                         groups=256, stride=1, padding=1),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Conv2d(256, 4 * 4,
                                                         kernel_size=1))

        self.confidence_layer5 = nn.Sequential(nn.Conv2d(256, 256,
                                                         kernel_size=3,
                                                         groups=256, stride=1, padding=1),
                                               nn.BatchNorm2d(256),
                                               nn.ReLU(),
                                               nn.Conv2d(256, 4 * self.num_classes,
                                                         kernel_size=1))

        self.extra4 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(64))

        self.regression_layer6 = nn.Sequential(nn.Conv2d(64, 64,
                                                         kernel_size=3,
                                                         groups=64, stride=1, padding=1),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 4 * 4,
                                                         kernel_size=1))

        self.confidence_layer6 = nn.Sequential(nn.Conv2d(64, 64,
                                                         kernel_size=3,
                                                         groups=64, stride=1, padding=1),
                                               nn.BatchNorm2d(64),
                                               nn.ReLU(),
                                               nn.Conv2d(64, 4 * self.num_classes,
                                                         kernel_size=1))

        if phase == 'test':
            self.softmax = nn.Softmax()
            self.inference_layer = InferenceLayer(top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def forward(self, x):
        loc_list = []
        class_list = []

        x = self.conv_layer1(x)
        x = self.invertedresidual_layer1(x)
        x = self.invertedresidual_layer2(x)
        x = self.invertedresidual_layer3(x)
        x = self.invertedresidual_layer4(x)
        x = self.invertedresidual_layer5(x)
        x = self.invertedresidual_layer6(x)
        x = self.invertedresidual_layer7(x)
        x = self.invertedresidual_layer8(x)
        x = self.invertedresidual_layer9(x)
        x = self.invertedresidual_layer10(x)
        x = self.invertedresidual_layer11(x)
        x = self.invertedresidual_layer12(x)
        x = self.invertedresidual_layer13(x)

        rl1 = self.regression_layer1(x)
        cl1 = self.confidence_layer1(x)

        loc_list.append(rl1)
        class_list.append(cl1)

        x = self.invertedresidual_layer14(x)
        x = self.invertedresidual_layer15(x)
        x = self.invertedresidual_layer16(x)
        x = self.invertedresidual_layer17(x)
        x = self.conv_layer2(x)

        rl2 = self.regression_layer2(x)
        cl2 = self.confidence_layer2(x)

        loc_list.append(rl2)
        class_list.append(cl2)

        x = self.extra1(x)
        rl3 = self.regression_layer3(x)
        cl3 = self.confidence_layer3(x)

        loc_list.append(rl3)
        class_list.append(cl3)

        x = self.extra2(x)
        rl4 = self.regression_layer4(x)
        cl4 = self.confidence_layer4(x)

        loc_list.append(rl4)
        class_list.append(cl4)

        x = self.extra3(x)
        rl5 = self.regression_layer5(x)
        cl5 = self.confidence_layer5(x)

        loc_list.append(rl5)
        class_list.append(cl5)

        x = self.extra4(x)
        rl6 = self.regression_layer6(x)
        cl6 = self.confidence_layer6(x)

        loc_list.append(rl6)
        class_list.append(cl6)

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
