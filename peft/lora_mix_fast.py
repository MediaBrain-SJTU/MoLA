"""
Implementation of LoRA (LoRA: Low-Rank Adaptation of Large Language Models: https://arxiv.org/abs/2106.09685)
Codes are modified from (https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import time

class LoRALayer(nn.Module):
    """
    Base lora class
    """
    def __init__(
            self,
            r,
            lora_alpha,
         ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Mark the weight as unmerged
        self.merged = False

    def reset_parameters(self):
        raise NotImplementedError

    def train(self, mode:bool = True):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError




class MultiLoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, n_expert):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer
        self.n_expert = n_expert

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))) for th in range(n_expert)]) 

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

        self.merged = False
        self.label_batch = None

    def reset_parameters(self):
        for th in range(self.n_expert):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def forward(self, x):
        for th in range(self.n_expert):
            self.conv.weight.data  = self.conv.weight.data + (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
        outputs = self.conv(x)
        return outputs
        

    def merged_weight(self, th):
        self.conv.weight.data += (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
        self.merged = True




class MultiLoraConv2dV2(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, n_expert):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer
        self.n_expert = n_expert

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))) for th in range(n_expert)]) 

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

        self.merged = False
        self.label_batch = None

    def reset_parameters(self):
        for th in range(self.n_expert):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def train(self, mode: bool = True):
        self.conv.train(mode)
        if self.merged:
            for th in range(self.n_expert):
                self.conv.weight.data  = self.conv.weight.data + (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        self.conv.eval()
        if not self.merged:
            for th in range(self.n_expert):
                self.conv.weight.data = self.conv.weight.data - (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if not self.merged:
            return F.conv2d(
                x,
                self.conv.weight + torch.sum(torch.stack([(self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling for th in range(self.n_expert)]),dim=0),
                self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
            )
        return self.conv(x)


class MultiExpertConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, n_expert):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer
        self.n_expert = n_expert

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]


        self.expert_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((out_channels * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 

        self.scaling = self.lora_alpha / self.r

        self.merged = False
        self.label_batch = None


    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def forward(self, x):
        for th in range(self.n_expert):
            self.conv.weight.data  = self.conv.weight.data + (self.expert_list[th]).view(self.conv.weight.shape)
        outputs = self.conv(x)
        return outputs
        

    def merged_weight(self, th):
        self.conv.weight.data += (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
        self.merged = True




class MultiExpertConv2dV2(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, n_expert):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer
        self.n_expert = n_expert

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.expert_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((out_channels * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 

        self.scaling = self.lora_alpha / self.r

        self.merged = False
        self.label_batch = None


    def train(self, mode: bool = True):
        self.conv.train(mode)
        if self.merged:
            for th in range(self.n_expert):
                self.conv.weight.data  = self.conv.weight.data + (self.expert_list[th]).view(self.conv.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        self.conv.eval()
        if not self.merged:
            for th in range(self.n_expert):
                self.conv.weight.data = self.conv.weight.data - (self.expert_list[th]).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if not self.merged:
            return F.conv2d(
                x,
                self.conv.weight + torch.sum(torch.stack([(self.expert_list[th]).view(self.conv.weight.shape) * self.scaling for th in range(self.n_expert)]),dim=0),
                self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
            )
        return self.conv(x)
