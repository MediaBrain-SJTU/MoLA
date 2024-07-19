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

from layers.moe_layer import MoEBase, LoraConv, SoftLoraConv
from layers.router import build_router
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, normal_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmpretrain.models.utils import channel_shuffle
from code.models.base_backbone import BaseBackbone

from code.models.shufflenet_v2 import ShuffleNetV2


class InvertedResidual(BaseModule):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.

    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f'in_channels ({in_channels}) should equal to '
                f'branch_features * 2 ({branch_features * 2}) '
                'when stride is 1')

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f'stride ({self.stride}) should not equal 1 when '
                f'in_channels != branch_features * 2')

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def forward(self, input_x):
        x, alphas = input_x[0], input_x[1]

        def _inner_forward(input_x):
            x, alphas = input_x[0], input_x[1]
            if self.stride > 1:
                branch1_out = self.branch1[0](x)
                branch1_out = self.branch1[1]({0:branch1_out, 1:alphas})
                branch2_out = self.branch2[0]({0:x, 1:alphas})
                branch2_out = self.branch2[1](branch2_out)
                branch2_out = self.branch2[2]({0:branch2_out, 1:alphas})
                out = torch.cat((branch1_out, branch2_out), dim=1)
            else:
                # Channel Split operation. using these lines of code to replace
                # ``chunk(x, 2, dim=1)`` can make it easier to deploy a
                # shufflenetv2 model by using mmdeploy.
                channels = x.shape[1]
                c = channels // 2 + channels % 2
                x1 = x[:, :c, :, :]
                x2 = x[:, c:, :, :]

                branch2_out = self.branch2[0]({0:x2, 1:alphas})
                branch2_out = self.branch2[1](branch2_out)
                branch2_out = self.branch2[2]({0:branch2_out, 1:alphas})

                out = torch.cat((x1, branch2_out), dim=1)

            out = channel_shuffle(out, 2)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, {0:x, 1:alphas})
        else:
            out = _inner_forward({0:x, 1:alphas})

        return out


class ShuffleNetV2Grad(BaseBackbone):
    def __init__(self,
                 widen_factor=1.0,
                 out_indices=(3, ),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=None):
        super(ShuffleNetV2Grad, self).__init__(init_cfg)
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError('widen_factor must be in [0.5, 1.0, 1.5, 2.0]. '
                             f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.out_dim = output_channels
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))

    def _make_layer(self, out_channels, num_blocks):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        super(ShuffleNetV2, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            return

        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'conv1' in name:
                    normal_init(m, mean=0, std=0.01)
                else:
                    normal_init(m, mean=0, std=1.0 / m.weight.shape[1])
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m.weight, val=1, bias=0.0001)
                if isinstance(m, _BatchNorm):
                    if m.running_mean is not None:
                        nn.init.constant_(m.running_mean, 0)

    def forward(self, x, task_label=None, return_intermediate_features=False, norm_flag=False):
        x = self.conv1({0:x, 1:task_label})
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate(self.layers):
            if layer.__class__ == ConvModule:
                x = layer({0:x, 1:task_label})
            else:
                for th, each_block in enumerate(layer):
                    x = each_block({0:x, 1:task_label})
            if i in self.out_indices:
                outs.append(x)

        if return_intermediate_features:
            return outs[-1], outs, None
        else:
            return outs[-1]

    def train(self, mode=True):
        super(ShuffleNetV2Grad, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


def shufflenet(pretrained=False, **kwargs):
    return ShuffleNetV2(widen_factor=1.0)


class ResNet18_LoraMix(ShuffleNetV2):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        super(ResNet18_LoraMix, self).__init__(widen_factor=1.0)

        from peft.lora_mix_fast_shufflenet import MultiLoraConv2dV2
        self.replace_lora_expert(MultiLoraConv2dV2, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.conv1.conv
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
        
        if target_conv.groups == 1:
            setattr(self.conv1, "conv", adapter)

        target_layers = self.layers
        for th, layer in enumerate(target_layers):
            if th == len(target_layers)-1:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=layer.conv, n_expert=n_expert)
                if layer.conv.groups == 1:
                    setattr(layer, 'conv', adapter)
                break
            for bottleneck_layer in layer:
                if hasattr(bottleneck_layer, 'branch1'):
                    for each_branch in [bottleneck_layer.branch1, bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)
                else:
                    for each_branch in [bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)


class ResNet18_LoraGrad(ShuffleNetV2Grad):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        super(ResNet18_LoraGrad, self).__init__(widen_factor=1.0)

        from peft.lora_fast_shufflenet import MultiLoraConv2d
        self.replace_lora_expert(MultiLoraConv2d, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.conv1.conv
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
        
        if target_conv.groups == 1:
            setattr(self.conv1, "conv", adapter)

        target_layers = self.layers
        for th, layer in enumerate(target_layers):
            if th == len(target_layers)-1:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=layer.conv, n_expert=n_expert)
                if layer.conv.groups == 1:
                    setattr(layer, 'conv', adapter)
                break
            for bottleneck_layer in layer:
                if hasattr(bottleneck_layer, 'branch1'):
                    for each_branch in [bottleneck_layer.branch1, bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)
                else:
                    for each_branch in [bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)


class ResNet_SoftRouter(MoEBase):
    def __init__(self, orig_resnet, dilate_scale=8, n_expert=0):
        super(ResNet_SoftRouter, self).__init__()

        self.conv1 = orig_resnet.conv1
        self.maxpool = orig_resnet.maxpool
        self.layers = orig_resnet.layers

        self.high_dim = 32
        self.map_to_high_dim = torch.nn.Linear(n_expert, self.high_dim)

    def forward(self, x, return_scores=False):
        if self.router is not None:
            router_score = self.router(x)
            self.set_score(router_score)
        x = self.conv1(x)
        x = self.maxpool(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        if return_scores:
            return x, self.map_to_high_dim(router_score)
        else:
            return x


class ResNet18_LoraSoftRouter(ResNet_SoftRouter):
    def __init__(self, pretrained=True, n_expert=0, gamma=0, lora_alpha=0):
        super(ResNet18_LoraSoftRouter, self).__init__(ShuffleNetV2(widen_factor=1.0), n_expert=n_expert)

        from layers.moe_layer import SoftLoraConvV2
        self.replace_lora_expert(SoftLoraConvV2, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.conv1.conv
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
        
        if target_conv.groups == 1:
            setattr(self.conv1, "conv", adapter)

        target_layers = self.layers
        for th, layer in enumerate(target_layers):
            if th == len(target_layers)-1:
                adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=layer.conv, n_expert=n_expert)
                if layer.conv.groups == 1:
                    setattr(layer, 'conv', adapter)
                break
            for bottleneck_layer in layer:
                if hasattr(bottleneck_layer, 'branch1'):
                    for each_branch in [bottleneck_layer.branch1, bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)
                else:
                    for each_branch in [bottleneck_layer.branch2]:
                        for each_conv in each_branch:
                            adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=each_conv.conv, n_expert=n_expert)
                            if each_conv.conv.groups == 1:
                                setattr(each_conv, 'conv', adapter)

