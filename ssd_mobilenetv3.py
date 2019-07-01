import torch
import torch.nn as nn
from utils.box_utils import priorbox
from layers.inference import InferenceLayer
from torch.autograd import Variable
import torch.nn.functional as F

class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * (self.relu6(x+3)) / 6

class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HardSigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return (self.relu6(x+3)) / 6

class SqueezeAndExcite(nn.Module):
    def __init__(self, n_features, reduction=4):
        super(SqueezeAndExcite, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU6(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = HardSigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class SSD_MobileNetV3(nn.Module):
        def __init__(self, phase='train', num_classes=2):
                super(SSD_MobileNetV3, self).__init__()
            
                self.phase = phase
                self.num_classes=num_classes
                self.priors = Variable(priorbox('mobilenetv3'), volatile=True)

                self.conv_layer1 = nn.Sequential(
                          nn.Conv2d(3, 16, 3, 2, 1, bias=False),
                          nn.BatchNorm2d(16),
                          HardSwish())

                '''
                input = 16
                kernel, expansion output_channel  se     nl  stride
                  3        16           16        True   'RE'   2
                  3        72           24        False  'RE'   2
                  3        88           24        False  'RE'   1
                  5        96           40        True   'HS'   2
                  5       240           40        True   'HS'   1
                  5       240           40        True   'HS'   1
                  5       120           48        True   'HS'   1
                  5       144           48        True   'HS'   1
                  5       288           96        True   'HS'   2
                  5       576           96        True   'HS'   1
                  5       576           96        True   'HS'   1
                '''
                
                self.invertedresidual_layer1 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, 2, 1, groups=16, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU6(inplace=True),
                        SqueezeAndExcite(16, reduction=4),
                        nn.Conv2d(16, 16, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(16)
                )

                self.invertedresidual_layer2 = nn.Sequential(
                        nn.Conv2d(16, 72, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(72),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(72, 72, 3, 2, 1, groups=72, bias=False),
                        nn.BatchNorm2d(72),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(72, 24, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(24)
                )

                # Special
                self.invertedresidual_layer3 = nn.Sequential(
                        nn.Conv2d(24, 88, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(88),
                        nn.ReLU6(inplace=True),
                        nn.Conv2d(88, 88, 3, 1, 1, groups=88, bias=False),
                        nn.BatchNorm2d(88),
                        nn.Conv2d(88, 24, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(24)
                )

                self.invertedresidual_layer4 = nn.Sequential(
                        nn.Conv2d(24, 96, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(96),
                        HardSwish(),
                        nn.Conv2d(96, 96, 5, 2, 2, groups=96, bias=False),
                        nn.BatchNorm2d(96),
                        HardSwish(),
                        SqueezeAndExcite(96, reduction=4),
                        nn.Conv2d(96, 40, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(40)
                )

                # Special
                self.invertedresidual_layer5 = nn.Sequential(
                        nn.Conv2d(40, 240, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(240),
                        HardSwish(),
                        nn.Conv2d(240, 240, 5, 1, 2, groups=240, bias=False),
                        nn.BatchNorm2d(240),
                        HardSwish(),
                        SqueezeAndExcite(240, reduction=4),
                        nn.Conv2d(240, 40, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(40)
                )

                # Special
                self.invertedresidual_layer6 = nn.Sequential(
                        nn.Conv2d(40, 240, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(240),
                        HardSwish(),
                        nn.Conv2d(240, 240, 5, 1, 2, groups=240, bias=False),
                        nn.BatchNorm2d(240),
                        HardSwish(),
                        SqueezeAndExcite(240, reduction=4),
                        nn.Conv2d(240, 40, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(40)
                ) 

                self.invertedresidual_layer7 = nn.Sequential(
                        nn.Conv2d(40, 120, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(120),
                        HardSwish(),
                        nn.Conv2d(120, 120, 5, 1, 2, groups=120, bias=False),
                        nn.BatchNorm2d(120),
                        HardSwish(),
                        SqueezeAndExcite(120, reduction=4),
                        nn.Conv2d(120, 48, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(48) 
                )

                # Special
                self.invertedresidual_layer8 = nn.Sequential(
                        nn.Conv2d(48, 144, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(144),
                        HardSwish(),
                        nn.Conv2d(144, 144, 5, 1, 2, groups=144, bias=False),
                        nn.BatchNorm2d(144),
                        HardSwish(),
                        SqueezeAndExcite(144, reduction=4),
                        nn.Conv2d(144, 48, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(48) 
                )

                self.regression_layer1 =  nn.Sequential(nn.Conv2d(48, 48, 
                                                        kernel_size=3,
                                                        groups=48, stride=1, padding=1),
                                                        nn.BatchNorm2d(48),
                                                        nn.ReLU(),
                                                        nn.Conv2d(48, 6*4, 
                                                        kernel_size=1))
                
                self.confidence_layer1 =  nn.Sequential(nn.Conv2d(48, 48, 
                                                        kernel_size=3,
                                                        groups=48, stride=1, padding=1),
                                                        nn.BatchNorm2d(48),
                                                        nn.ReLU(),
                                                        nn.Conv2d(48, 6*self.num_classes, 
                                                        kernel_size=1))

                self.invertedresidual_layer9 = nn.Sequential(
                        nn.Conv2d(48, 288, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(288),
                        HardSwish(),
                        nn.Conv2d(288, 288, 5, 2, 2, groups=288, bias=False),
                        nn.BatchNorm2d(288),
                        HardSwish(),
                        SqueezeAndExcite(288, reduction=4),
                        nn.Conv2d(288, 96, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(96) 
                )

                # Special
                self.invertedresidual_layer10 = nn.Sequential(
                        nn.Conv2d(96, 576, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(576),
                        HardSwish(),
                        nn.Conv2d(576, 576, 5, 1, 2, groups=576, bias=False),
                        nn.BatchNorm2d(576),
                        HardSwish(),
                        SqueezeAndExcite(576, reduction=4),
                        nn.Conv2d(576, 96, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(96) 
                )

                # Special
                self.invertedresidual_layer11 = nn.Sequential(
                        nn.Conv2d(96, 576, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(576),
                        HardSwish(),
                        nn.Conv2d(576, 576, 5, 1, 2, groups=576, bias=False),
                        nn.BatchNorm2d(576),
                        HardSwish(),
                        SqueezeAndExcite(576, reduction=4),
                        nn.Conv2d(576, 96, 1, 1, 0, bias=False),
                        nn.BatchNorm2d(96) 
                )

                self.conv_layer2 = nn.Sequential(
                        nn.Conv2d(96, 576, 1, 1, 0, bias=False),
                        SqueezeAndExcite(576, reduction=4),
                        nn.BatchNorm2d(576),
                        HardSwish()
                )
                
                self.conv_layer3 = nn.Sequential(
                        nn.Conv2d(576, 1280, 1, 1, 0, bias=False),
                        SqueezeAndExcite(1280, reduction=4),
                        nn.BatchNorm2d(1280),
                        HardSwish()
                )

                self.regression_layer2 =  nn.Sequential(nn.Conv2d(1280, 1280, 
                                                        kernel_size=3,
                                                        groups=1280, stride=1, padding=1),
                                                        nn.BatchNorm2d(1280),
                                                        nn.ReLU(),
                                                        nn.Conv2d(1280, 6*4, 
                                                        kernel_size=1))

                self.confidence_layer2 =  nn.Sequential(nn.Conv2d(1280, 1280, 
                                                        kernel_size=3,
                                                        groups=1280, stride=1, padding=1),
                                                        nn.BatchNorm2d(1280),
                                                        nn.ReLU(),
                                                        nn.Conv2d(1280, 6*self.num_classes, 
                                                        kernel_size=1))

                self.extra1 = nn.Sequential(nn.Conv2d(1280, 256, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(256, 512, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(512))

                self.regression_layer3 =  nn.Sequential(nn.Conv2d(512, 512,
                                                        kernel_size=3,
                                                        groups=512, stride=1, padding=1),
                                                        nn.BatchNorm2d(512),
                                                        nn.ReLU(),
                                                        nn.Conv2d(512, 6*4,
                                                        kernel_size=1))
                
                self.confidence_layer3 =  nn.Sequential(nn.Conv2d(512, 512,
                                                        kernel_size=3,
                                                        groups=512, stride=1, padding=1),
                                                        nn.BatchNorm2d(512),
                                                        nn.ReLU(),
                                                        nn.Conv2d(512, 6*self.num_classes,
                                                        kernel_size=1))

                self.extra2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(256))

                self.regression_layer4 =  nn.Sequential(nn.Conv2d(256, 256,
                                                        kernel_size=3,
                                                        groups=256, stride=1, padding=1),
                                                        nn.BatchNorm2d(256),
                                                        nn.ReLU(),
                                                        nn.Conv2d(256, 4*4,
                                                        kernel_size=1))

                self.confidence_layer4 =  nn.Sequential(nn.Conv2d(256, 256,
                                                        kernel_size=3,
                                                        groups=256, stride=1, padding=1),
                                                        nn.BatchNorm2d(256),
                                                        nn.ReLU(),
                                                        nn.Conv2d(256, 4*self.num_classes,
                                                        kernel_size=1))
                
                self.extra3 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(256))

                self.regression_layer5 =  nn.Sequential(nn.Conv2d(256, 256,
                                                        kernel_size=3,
                                                        groups=256, stride=1, padding=1),
                                                        nn.BatchNorm2d(256),
                                                        nn.ReLU(),
                                                        nn.Conv2d(256, 4*4,
                                                        kernel_size=1))

                self.confidence_layer5 =  nn.Sequential(nn.Conv2d(256, 256,
                                                        kernel_size=3,
                                                        groups=256, stride=1, padding=1),
                                                        nn.BatchNorm2d(256),
                                                        nn.ReLU(),
                                                        nn.Conv2d(256, 4*self.num_classes,
                                                        kernel_size=1))
                
                self.extra4 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(64, 64, 1, 1, 0, bias=False),
                                            nn.BatchNorm2d(64))

                self.regression_layer6 =  nn.Sequential(nn.Conv2d(64, 64,
                                                        kernel_size=3,
                                                        groups=64, stride=1, padding=1),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(),
                                                        nn.Conv2d(64, 4*4,
                                                        kernel_size=1))

                self.confidence_layer6 =  nn.Sequential(nn.Conv2d(64, 64,
                                                        kernel_size=3,
                                                        groups=64, stride=1, padding=1),
                                                        nn.BatchNorm2d(64),
                                                        nn.ReLU(),
                                                        nn.Conv2d(64, 4*self.num_classes,
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
            x = x + self.invertedresidual_layer3(x)
            x = self.invertedresidual_layer4(x)
            x = x + self.invertedresidual_layer5(x)
            x = x + self.invertedresidual_layer6(x)
            x = self.invertedresidual_layer7(x)
            x = x + self.invertedresidual_layer8(x)
            
            rl1 = self.regression_layer1(x)
            cl1 = self.confidence_layer1(x)
            loc_list.append(rl1)
            class_list.append(cl1)


            x = self.invertedresidual_layer9(x)
            x = x + self.invertedresidual_layer10(x)
            x = x + self.invertedresidual_layer11(x) 
            x = self.conv_layer2(x)
            x = self.conv_layer3(x)
            
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

