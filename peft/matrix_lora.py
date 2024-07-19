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


class LoRALinear(LoRALayer):
    def __init__(self, r, lora_alpha, linear_layer):
        """
        LoRA class for nn.Linear class
        :param r: low rank dimension
        :param lora_alpha: scaling factor
        :param linear_layer: target nn.Linear layer for applying Lora
        """
        super().__init__(r, lora_alpha)
        self.linear = linear_layer

        in_features = self.linear.in_features
        out_features = self.linear.out_features

        # Lora configuration
        self.lora_A = nn.Parameter(self.linear.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.linear.weight.new_zeros((out_features, r)))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


    def train(self, mode:bool = True):
        self.linear.train(mode)
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False


    def eval(self):
        self.linear.eval()
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True


    def forward(self, x):
        if not self.merged:
            result = F.linear(x, self.linear.weight, bias=self.linear.bias)
            out = (x @ self.lora_A.T @ self.lora_B.T)
            result += out
            return result
        else:
            return F.linear(x, self.linear.weight, bias=self.linear.bias)


class LoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.lora_A = nn.Parameter(
            self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))
        )
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.conv.train(mode)
        if self.merged:
            # Make sure that the weights are not merged
            self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        self.conv.eval()
        if not self.merged:
            # Merge the weights and mark it
            self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x):
        if not self.merged:
            return F.conv2d(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
            )
        return self.conv(x)




class MultiLoraConv2d(LoRALayer):
    def __init__(self, r, lora_alpha, conv_layer, num_task):
        """
        LoRA class for nn.Conv2d class
        """
        super().__init__(r, lora_alpha)
        self.conv = conv_layer
        self.num_task = num_task

        in_channels = self.conv.in_channels
        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size[0]

        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))) for th in range(num_task)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(self.conv.weight.new_zeros((out_channels * kernel_size, r * kernel_size))) for th in range(num_task)]) 

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

        self.conv_weight_stack = nn.Parameter(torch.cat([(self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling for th in range(self.num_task)], dim=0))

        self.merged = False
        self.label_batch = None

    def reset_parameters(self):
        for th in range(self.num_task):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def train(self, mode: bool = True):
        self.conv.train(mode)

    def eval(self):
        self.conv.eval()

    def forward(self, x):
        if not self.merged:

            if x.shape[1] == self.conv_weight_stack.shape[1]:
                x = x.view(-1, x.shape[1]*(self.num_task+1), x.shape[2], x.shape[3])

            batch_size, multi_c = x.shape[0], x.shape[1]

            outputs1 = F.conv2d(x[:,:multi_c//(self.num_task+1)], self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
            outputs2 = F.conv2d(x[:,multi_c//(self.num_task+1):], self.conv_weight_stack, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, groups=self.num_task)
            outputs1 = outputs1.view(batch_size,-1,outputs1.shape[2],outputs1.shape[3])
            outputs2 = outputs2.view(batch_size*self.num_task,-1,outputs2.shape[2],outputs2.shape[3])
            outputs = torch.cat((outputs1, outputs2), dim=0)
            return outputs
        else:
            return self.conv(x)

    def merged_weight(self, th):
        self.conv.weight.data += (self.lora_B_list[th] @ self.lora_A_list[th]).view(self.conv.weight.shape) * self.scaling
        self.merged = True

