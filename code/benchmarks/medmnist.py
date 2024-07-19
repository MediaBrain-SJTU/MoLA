import torch
import torch.nn.functional as F
import torch.nn as nn

from . import mtl_benchmark

from code.models.medmnist import (
    ResNet18,
    ResNet18_LoraMix,
    ResNet18_LoraGrad,
    ResNet18_LoraRouter,
    ResNet18_LoraSoftRouter,
)

from code.models.segnet_mtan import MTANEncoder
from code.data.datasets import MedData
from code.evaluation.medmnist import MedEvaluator

from scl_loss import moe_cl_loss
from code.models.cross_stitch import Cross_stitch
from code.models.MMoE import MMoE
from code.models.DSelect_k import DSelect_k
from code.models.HPS import HPS
from code.models.CGC import CGC
from code.models.LTB import LTB
from code.models.MTAN_18 import MTAN


class MedModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet18(True)
        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = self.encoder.layer4


class MedModel_LoRAGrad(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet18_LoraGrad(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha)
        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = self.encoder.layer4


class MedModel_LoRASoftRouter(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet18_LoraSoftRouter(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha)
        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = self.encoder.layer4


class MedModel_CrossStitch(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}

        self.encoder = Cross_stitch(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_MMoE(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224) 
        kwargs['arch_args']['num_experts'] = [args.n_expert]

        self.encoder = MMoE(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_DSelect_k(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224) 
        kwargs['arch_args']['num_experts'] = [args.n_expert]
        kwargs['arch_args']['kgamma'] = 1.0 
        kwargs['arch_args']['num_nonzeros'] = 2

        self.encoder = DSelect_k(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_LTB(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}

        self.encoder = LTB(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_CGC(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224) 
        kwargs['arch_args']['num_experts'] = [1]+[1]*9 

        self.encoder = CGC(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_HPS(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}

        self.encoder = HPS(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


class MedModel_MTAN(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        def encoder_class(): 
            return ResNet18(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}

        self.encoder = MTAN(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["0"] = nn.Linear(512, 9)
        self.decoders["1"] = nn.Linear(512, 4)
        self.decoders["2"] = nn.Linear(512, 11)
        self.decoders["3"] = nn.Linear(512, 8)
        self.decoders["4"] = nn.Linear(512, 2)
        self.decoders["5"] = nn.Linear(512, 7)
        self.decoders["6"] = nn.Linear(512, 11)
        self.decoders["7"] = nn.Linear(512, 11)
        self.decoders["8"] = nn.Linear(512, 2)
        self.last_shared_layer = None


@mtl_benchmark.register("medmnist_resnet18")
class MedBenchmark(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
            task_criteria={
                "0": torch.nn.CrossEntropyLoss(),
                "1": torch.nn.CrossEntropyLoss(),
                "2": torch.nn.CrossEntropyLoss(),
                "3": torch.nn.CrossEntropyLoss(),
                "4": torch.nn.CrossEntropyLoss(),
                "5": torch.nn.CrossEntropyLoss(),
                "6": torch.nn.CrossEntropyLoss(),
                "7": torch.nn.CrossEntropyLoss(),
                "8": torch.nn.CrossEntropyLoss(),
            }
        )
        self.datasets = {
            'train': MedData(split='train'),
            'valid': MedData(split='test')
        }

    def evaluate(self, model, loader):
        return MedEvaluator.evaluate(model, loader, device=next(model.parameters()).device)

    def evaluate_task(self, model, loader):
        return MedEvaluator.evaluate_task(model, loader, device=next(model.parameters()).device)

    def evaluate_each_task(self, model, loader):
        return MedEvaluator.evaluate_each_task(model, loader, device=next(model.parameters()).device)


    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=128, test_batch=128, epochs=100, lr=0.05, n_expert=9)
        return parser

    @staticmethod
    def get_model(args):
        if args.arch == 'lora_soft_router':
            return MedModel_LoRASoftRouter(args)
        elif args.arch == 'lora_grad':
            return MedModel_LoRAGrad(args)
        elif args.arch == 'Cross_stitch':
            return MedModel_CrossStitch(args)
        elif args.arch == 'MMoE':
            return MedModel_MMoE(args)
        elif args.arch == 'DSelect_k':
            return MedModel_DSelect_k(args)
        elif args.arch == 'LTB':
            return MedModel_LTB(args)
        elif args.arch == 'CGC':
            return MedModel_CGC(args)
        elif args.arch == 'PLE':
            return MedModel_PLE(args)
        elif args.arch == 'HPS':
            return MedModel_HPS(args)
        elif args.arch == 'MTAN':
            return MedModel_MTAN(args)
        else:
            return MedModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        

