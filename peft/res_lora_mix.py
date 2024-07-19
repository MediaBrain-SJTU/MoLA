import torch
import torch.nn as nn
from peft.lora_mix_fast import MultiLoraConv2d

class Res_Lora_Mix(nn.Module):
    def __init__(self, resnet, n_expert=5, gamma=4, lora_alpha=16, **kwargs):
        super(Res_Lora_Mix, self).__init__()
        self.resnet = resnet
        self.n_expert = n_expert
        self.gamma = gamma
        self.lora_alpha = lora_alpha
        self.replace_lora_expert(MultiLoraConv2d, n_expert, gamma, lora_alpha)

    def replace_lora_expert(self, adapter_class, n_expert, gamma, lora_alpha):
        target_conv = self.resnet.conv1
        adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
        
        setattr(self.resnet, "conv1", adapter)
        target_layers = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        ]
        for layer in target_layers:
            for bottleneck_layer in layer:
                for cv in ["conv1", "conv2"]:
                    target_conv = getattr(bottleneck_layer, cv)
                    adapter = adapter_class(r=gamma, lora_alpha=lora_alpha, conv_layer=target_conv, n_expert=n_expert)
                    setattr(bottleneck_layer, cv, adapter)

    def forward(self, x, return_features=False):
        return self.resnet(x, return_features)