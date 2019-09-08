import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from models.efficientnet_backbone_utils import *
from utils.box_utils import priorbox
from layers.inference import InferenceLayer


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class SSD_EfficientNet(nn.Module):
    def __init__(self, num_classes, phase='train', blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        self.phase = phase
        self.num_classes = num_classes
        self.priors = Variable(priorbox('efficientnet'), volatile=True)
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self.regression_layer1 = nn.Sequential(nn.Conv2d(160, 160,
                                                         kernel_size=3,
                                                         groups=160, stride=1, padding=1),
                                               nn.BatchNorm2d(160),
                                               nn.ReLU(),
                                               nn.Conv2d(160, 6 * 4,
                                                         kernel_size=1))

        self.confidence_layer1 = nn.Sequential(nn.Conv2d(160, 160,
                                                         kernel_size=3,
                                                         groups=160, stride=1, padding=1),
                                               nn.BatchNorm2d(160),
                                               nn.ReLU(),
                                               nn.Conv2d(160, 6 * self.num_classes,
                                                         kernel_size=1))

        self.regression_layer2 = nn.Sequential(nn.Conv2d(1792, 1792,
                                                         kernel_size=3,
                                                         groups=1792, stride=1, padding=1),
                                               nn.BatchNorm2d(1792),
                                               nn.ReLU(),
                                               nn.Conv2d(1792, 6 * 4,
                                                         kernel_size=1))

        self.confidence_layer2 = nn.Sequential(nn.Conv2d(1792, 1792,
                                                         kernel_size=3,
                                                         groups=1792, stride=1, padding=1),
                                               nn.BatchNorm2d(1792),
                                               nn.ReLU(),
                                               nn.Conv2d(1792, 6 * self.num_classes,
                                                         kernel_size=1))

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

        self.extra1 = nn.Sequential(nn.Conv2d(1792, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, 3, 2, 1, groups=256, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 512, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(512))

        self.extra2 = nn.Sequential(nn.Conv2d(512, 128, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256))

        self.extra3 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, 3, 2, 1, groups=128, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 256, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(256))

        self.extra4 = nn.Sequential(nn.Conv2d(256, 64, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 3, 2, 1, groups=64, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(64))

        if self.phase == 'test':
            self.softmax = nn.Softmax()
            self.inference_layer = InferenceLayer(top_k=200, conf_thresh=0.01, nms_thresh=0.45)

    def extract_features(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate)
            if idx == 21:
                self.layer1 = x

        return self.layer1, x

    def forward(self, inputs):
        loc_list = []
        class_list = []
        layer1, x = self.extract_features(inputs)
        x = relu_fn(self._bn1(self._conv_head(x)))

        rl1 = self.regression_layer1(layer1)
        cl1 = self.confidence_layer1(layer1)

        loc_list.append(rl1)
        class_list.append(cl1)

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

    @classmethod
    def from_name(cls, model_name, num_classes=2, phase='train', override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return SSD_EfficientNet(num_classes, phase, blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b' + str(i) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

