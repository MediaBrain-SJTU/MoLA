import math
import os
import sys
from turtle import forward

import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from collections import OrderedDict
from peft.res_lora_mix import Res_Lora_Mix
from layers.moe_layer import MoEBase, LoraConv, SoftLoraConv
from layers.router import build_router
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


model_urls = {
   'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}



def load_url(url, model_dir="./pretrained", map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split("/")[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)


def conv3x3(in_planes, out_planes, conv_layer, stride=1, groups=1, dilation=1, **kwargs):
    """3x3 convolution with padding"""
    return conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=dilation, groups=groups, bias=False, dilation=dilation, **kwargs)


def conv1x1(in_planes, out_planes, conv_layer, stride=1, **kwargs):
    """1x1 convolution"""
    return conv_layer(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False, dilation=1, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlockMOE(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(BasicBlockMOE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, conv_layer, stride, **kwargs)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer, **kwargs)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockGrad(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, is_last=False):
        super(BasicBlockGrad, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, conv_layer, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, conv_layer)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, input_x):
        x, task_label = input_x[0], input_x[1]
        identity = x

        out = self.conv1(x, task_label)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, task_label)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return {0:out, 1:task_label}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(planes))
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(int(planes))
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(planes) * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckMOE(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(BottleneckMOE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_layer, **kwargs)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, conv_layer, stride, groups, dilation, **kwargs)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_layer, **kwargs)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class BottleneckGrad(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, conv_layer, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, is_last=False):
        super(BottleneckGrad, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_layer)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, conv_layer, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_layer)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, input_x):
        x, task_label = input_x[0], input_x[1]
        identity = x

        out = self.conv1(x, task_label)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, task_label)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out, task_label)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        preact = out
        out = self.relu(out)

        if self.is_last:
            return out, preact
        else:
            return {0:out, 1:task_label}


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.feature_dim = 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(int(planes) * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetMOE(MoEBase):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, n_expert=5, gamma=4, lora_alpha=16, ratio=1.0):
        super(ResNetMOE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_layer = LoraConv
        self.ratio = ratio
        self.normalize = None

        self.inplanes = int(ratio * 64)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckMOE):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, **kwargs):
        planes = int(self.ratio * planes)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.conv_layer, stride, **kwargs),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.conv_layer, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.conv_layer, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, **kwargs))

        return nn.Sequential(*layers)


class ResNetSoftMOE(MoEBase):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, n_expert=5, gamma=4, lora_alpha=16, ratio=1.0):
        super(ResNetSoftMOE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_layer = SoftLoraConv
        self.ratio = ratio
        self.normalize = None

        self.inplanes = int(ratio * 64)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], n_expert=n_expert, r=gamma, lora_alpha=lora_alpha)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BottleneckMOE):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, **kwargs):
        planes = int(self.ratio * planes)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.conv_layer, stride, **kwargs),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.conv_layer, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.conv_layer, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, **kwargs))

        return nn.Sequential(*layers)


class ResNetGrad(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None, ratio=1, **kwargs):
        super(ResNetGrad, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv_layer = nn.Conv2d
        self.ratio = ratio
        self.normalize = None

        self.inplanes = int(ratio * 64)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(512 * self.ratio) * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        planes = int(self.ratio * planes)
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, self.conv_layer, stride),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, self.conv_layer, stride, downsample, self.groups, self.base_width,
                            previous_dilation, norm_layer, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.conv_layer, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer, is_last=(_ == blocks - 1)))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, task_label=None, return_features=False):
        if self.normalize is not None:
            x = self.normalize(x)
        x = self.conv1(x, task_label)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = self.maxpool(x)

        x2, x2_pre = self.layer1({0:x1, 1:task_label})
        x3, x3_pre = self.layer2({0:x2, 1:task_label})
        x4, x4_pre = self.layer3({0:x3, 1:task_label})
        x5, x5_pre = self.layer4({0:x4, 1:task_label})

        x5 = self.avgpool(x5)
        x5 = torch.flatten(x5, 1)
        x6 = self.fc(x5)

        if return_features:
            return x6, x5
        else:
            return x6


    def forward(self, x, task_label=None, return_features=False):
        return self._forward_impl(x, task_label, return_features)


class ResNet_Router(MoEBase):
    def __init__(self, orig_resnet, dilate_scale=8, n_expert=0):
        super(ResNet_Router, self).__init__()

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x):
        if self.router is not None:
            self.set_score(self.router(x))
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet_SoftRouter(MoEBase):
    def __init__(self, orig_resnet, dilate_scale=8, n_expert=0):
        super(ResNet_SoftRouter, self).__init__()

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

        self.high_dim = 32
        self.map_to_high_dim = torch.nn.Linear(n_expert, self.high_dim)

    def forward(self, x, return_scores=False):
        if self.router is not None:
            router_score = self.router(x)
            self.set_score(router_score)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if return_scores:
            return x, self.map_to_high_dim(router_score)
        else:
            return x


class ResNet_LoRAGrad(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResNet_LoRAGrad, self).__init__()

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, task_label=None):
        x = self.relu1(self.bn1(self.conv1(x, task_label)))
        x1 = self.maxpool(x)

        x2, x2_pre = self.layer1({0:x1, 1:task_label})
        x3, x3_pre = self.layer2({0:x2, 1:task_label})
        x4, x4_pre = self.layer3({0:x3, 1:task_label})
        x5, x5_pre = self.layer4({0:x4, 1:task_label})

        return x5


class ResNet18(ResNet):
    def __init__(self, pretrained=True, block=BasicBlock, layers=[2,2,2,2]):
        super(ResNet18, self).__init__(block=block, layers=layers)
        if pretrained:
            self.load_state_dict(load_url(model_urls["resnet18"]), strict=False)

class ResNet18_LoraMix(ResNet):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        super(ResNet18_LoraMix, self).__init__(block=BasicBlock, layers=[2,2,2,2])
        if pretrained:
            self.load_state_dict(load_url(model_urls["resnet18"]), strict=False)

        from peft.lora_mix_fast import MultiLoraConv2dV2
        self.replace_lora_expert(MultiLoraConv2dV2, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.conv1
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
        
        setattr(self, "conv1", adapter)
        target_layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2", 'conv3']:
                    try:
                        target_conv = getattr(bottleneck_layer, cv)
                        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
                        setattr(bottleneck_layer, cv, adapter)
                    except:
                        continue

class ResNet18_LoraGrad(ResNet_LoRAGrad):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        resnet18 = ResNetGrad(BasicBlockGrad, [2,2,2,2])
        if pretrained:
            resnet18.load_state_dict(load_url(model_urls["resnet18"]), strict=False)
        super(ResNet18_LoraGrad, self).__init__(resnet18)

        from peft.lora_fast import MultiLoraConv2dV2
        self.replace_lora_expert(MultiLoraConv2dV2, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.conv1
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=n_expert)
        
        setattr(self, "conv1", adapter)
        target_layers = [
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2",'conv3']:
                    try:
                        target_conv = getattr(bottleneck_layer, cv)
                        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, num_task=n_expert)
                        setattr(bottleneck_layer, cv, adapter)
                    except:
                        continue


class ResNet18_LoraRouter(ResNet_Router):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        resnet18 = ResNetMOE(BasicBlockMOE, [2,2,2,2], n_expert=n_expert, gamma=gamma, lora_alpha=lora_alpha)
        if pretrained:
            resnet18.load_state_dict(load_url(model_urls["resnet18"]), strict=False)
        super(ResNet18_LoraRouter, self).__init__(resnet18, n_expert=n_expert)


class ResNet18_LoraSoftRouter(ResNet_SoftRouter):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        resnet18 = ResNetSoftMOE(BasicBlockMOE, [2,2,2,2], n_expert=n_expert, gamma=gamma, lora_alpha=lora_alpha)
        if pretrained:
            resnet18.load_state_dict(load_url(model_urls["resnet18"]), strict=False)
        super(ResNet18_LoraSoftRouter, self).__init__(resnet18, n_expert=n_expert)

