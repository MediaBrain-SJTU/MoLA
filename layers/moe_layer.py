import torch
from torch import autograd, nn as nn
import math
import torch.nn.functional as F

class GetMask(autograd.Function):
    @staticmethod
    def forward(ctx, scores):  # binarization

        expert_pred = torch.argmax(scores, dim=1)  # [bs]
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)

        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2

class GetSoftMask(autograd.Function):
    @staticmethod
    def forward(ctx, scores):  # binarization

        expert_pred = torch.argmax(scores, dim=1)  # [bs]
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)

        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"

def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"


class MoEBase(nn.Module):
    def __init__(self):
        super(MoEBase, self).__init__()
        self.scores = None
        self.router = None

    def set_score(self, scores):
        self.scores = scores
        for module in self.modules():
            if hasattr(module, 'scores'):
                module.scores = self.scores


class MoEConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5):
        super(MoEConv, self).__init__(in_channels, out_channels * n_expert, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels * n_expert
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1
        self.layer_selection = torch.zeros([n_expert, self.out_channels])
        for cluster_id in range(n_expert):
            start = cluster_id * self.expert_width
            end = (cluster_id + 1) * self.expert_width
            idx = torch.arange(start, end)
            self.layer_selection[cluster_id][idx] = 1
        self.scores = None

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            expert_selection, expert_selection_one_hot = GetMask.apply(self.scores)
            mask = torch.matmul(expert_selection_one_hot, self.layer_selection.to(x))  # [bs, self.out_channels]
            out = super(MoEConv, self).forward(x)
            out = out * mask.unsqueeze(-1).unsqueeze(-1)
            index = torch.where(mask.view(-1) > 0)[0]
            shape = out.shape
            out_selected = out.view(shape[0] * shape[1], shape[2], shape[3])[index].view(shape[0], -1, shape[2],
                                                                                         shape[3])
        else:
            out_selected = super(MoEConv, self).forward(x)
        self.scores = None
        return out_selected


class LoraConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5, r=4, lora_alpha=16):
        super(LoraConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1

        #################################
        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size))) for th in range(n_expert)]) 
        self.scaling = lora_alpha / r
        self.reset_lora_parameters(n_expert)
        #################################

        self.scores = None

    def reset_lora_parameters(self, n_expert):
        for th in range(n_expert):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            expert_selection, expert_selection_one_hot = GetMask.apply(self.scores)
            #################################
            conv_weight_stack = torch.cat([(self.lora_B_list[th] @ self.lora_A_list[th]).view(self.weight.shape).unsqueeze(0) * self.scaling for th in range(self.n_expert)], dim=0)
            batch_size, c = x.shape[0], x.shape[1]
            agg_weights = self.weight + torch.sum(
                torch.mul(conv_weight_stack.unsqueeze(0), expert_selection_one_hot.view(batch_size, -1, 1, 1, 1, 1)), dim=1)
            agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])
            x_grouped = x.view(1, -1, *x.shape[-2:])

            out = F.conv2d(x_grouped, agg_weights, self.bias, self.stride, self.padding, self.dilation, groups=batch_size)
            out_selected = out.view(batch_size, -1, *out.shape[-2:])
            #################################
        else:
            out_selected = super(LoraConv, self).forward(x)
        self.scores = None
        return out_selected




class SoftLoraConv(nn.Conv2d, MoEBase):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5, r=4, lora_alpha=16):
        super(SoftLoraConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1

        #################################
        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size))) for th in range(n_expert)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size))) for th in range(n_expert)]) 
        self.scaling = lora_alpha / r
        self.reset_lora_parameters(n_expert)
        #################################
        self.scores = None

    def reset_lora_parameters(self, n_expert):
        for th in range(n_expert):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            #################################
            gate_weight = F.softmax(self.scores, dim=1)
            batch_size, c = x.shape[0], x.shape[1]
            conv_weight_stack = torch.cat([(self.lora_B_list[th] @ self.lora_A_list[th]).view(self.weight.shape) * self.scaling for th in range(self.n_expert)], dim=0)
            agg_weights = torch.cat([self.weight, conv_weight_stack], dim=0)
            outs = F.conv2d(x, agg_weights, self.bias, self.stride, self.padding, self.dilation, groups=self.groups)
            shared_out = outs[:, :self.out_channels]
            experts_out = outs[:, self.out_channels:].view(batch_size, self.n_expert, self.out_channels, shared_out.shape[2], shared_out.shape[3])
            out_selected = shared_out + torch.sum(torch.mul(gate_weight.view(gate_weight.shape[0], gate_weight.shape[1], 1,1,1), experts_out), dim=1).squeeze()
            #################################

        else:
            out_selected = super(LoraConv, self).forward(x)
        self.scores = None
        return out_selected





class SoftLoraConvV2(nn.Conv2d, MoEBase):
    def __init__(self, r, lora_alpha, conv_layer, n_expert):
        super(SoftLoraConvV2, self).__init__(conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size, conv_layer.stride, conv_layer.padding, conv_layer.dilation,
            conv_layer.groups, conv_layer.bias, )
        # self.conv = conv_layer
        self.in_channels = conv_layer.in_channels
        self.out_channels = conv_layer.out_channels
        self.expert_width = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]

        self.n_expert = n_expert
        assert self.n_expert >= 1

        #################################
        # lora configuration
        self.lora_A_list = nn.ParameterList([nn.Parameter(conv_layer.weight.new_zeros((r * kernel_size, self.in_channels * kernel_size))) for th in range(n_expert)]) 
        self.lora_B_list = nn.ParameterList([nn.Parameter(conv_layer.weight.new_zeros((self.out_channels * kernel_size, r * kernel_size))) for th in range(n_expert)]) 
        self.scaling = lora_alpha / r
        self.reset_lora_parameters(n_expert)
        #################################
        self.scores = None

    def reset_lora_parameters(self, n_expert):
        for th in range(n_expert):
            nn.init.kaiming_uniform_(self.lora_A_list[th], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[th])

    def forward(self, x):
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)
            #################################
            gate_weight = F.softmax(self.scores, dim=1)
            batch_size, c = x.shape[0], x.shape[1]
            conv_weight_stack = torch.cat([(self.lora_B_list[th] @ self.lora_A_list[th]).view(self.weight.shape) * self.scaling for th in range(self.n_expert)], dim=0)
            agg_weights = torch.cat([self.weight, conv_weight_stack], dim=0)
            outs = F.conv2d(x, agg_weights, self.bias, self.stride, self.padding, self.dilation, groups=self.groups)
            shared_out = outs[:, :self.out_channels]
            experts_out = outs[:, self.out_channels:].view(batch_size, self.n_expert, self.out_channels, shared_out.shape[2], shared_out.shape[3])
            out_selected = shared_out + torch.sum(torch.mul(gate_weight.view(gate_weight.shape[0], gate_weight.shape[1], 1,1,1), experts_out), dim=1).squeeze()
            #################################

        else:
            out_selected = super(LoraConv, self).forward(x)
        self.scores = None
        return out_selected

