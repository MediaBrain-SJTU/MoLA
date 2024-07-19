import torch
import torch.nn.functional as F
import torch.nn as nn

from . import mtl_benchmark

from code.models.officehome import (
    ResNet50,
    ResNet50_LoraMix,
    ResNet50_LoraGrad,
    ResNet50_LoraRouter,
    ResNet50_LoraSoftRouter,
)

from code.models.segnet_mtan import MTANEncoder
from code.data.datasets import OfficeHome
from code.evaluation.officehome import OfficeHomeEvaluator

from scl_loss import moe_cl_loss
from code.models.cross_stitch import Cross_stitch
from code.models.MMoE import MMoE
from code.models.DSelect_k import DSelect_k
from code.models.HPS import HPS
from code.models.CGC import CGC
from code.models.LTB import LTB
from code.models.MTAN import MTAN


class OfficeHomeModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50(True)
        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = self.encoder.layer4


class OfficeHomeModel_LoRAGrad(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet50_LoraGrad(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha, lora_layer=args.lora_layer, lora_rank=args.lora_rank)
        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = self.encoder.layer4


class OfficeHomeModel_LoRASoftRouter(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet50_LoraSoftRouter(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha, lora_layer=args.lora_layer, lora_rank=args.lora_rank)
        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = self.encoder.layer4


class OfficeHomeModel_CrossStitch(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
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

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_MMoE(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224) 
        kwargs['arch_args']['num_experts'] = [4]

        self.encoder = MMoE(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_DSelect_k(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224)
        kwargs['arch_args']['num_experts'] = [4]
        kwargs['arch_args']['kgamma'] = 1.0 
        kwargs['arch_args']['num_nonzeros'] = 2

        self.encoder = DSelect_k(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_LTB(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
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

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_CGC(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 224, 224)
        kwargs['arch_args']['num_experts'] = [1,1,1,1,1] 

        self.encoder = CGC(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_HPS(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
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

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


class OfficeHomeModel_MTAN(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['P', 'A', 'C', 'R']
        def encoder_class(): 
            return ResNet50(True)
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

        self.decoders["P"] = nn.Linear(2048, 65)
        self.decoders["A"] = nn.Linear(2048, 65)
        self.decoders["C"] = nn.Linear(2048, 65)
        self.decoders["R"] = nn.Linear(2048, 65)
        self.last_shared_layer = None


@mtl_benchmark.register("officehome_resnet50")
class OfficeHomeBenchmark(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names = ['P', 'A', 'C', 'R'],
            task_criteria={
                "P": torch.nn.CrossEntropyLoss(),
                "A": torch.nn.CrossEntropyLoss(),
                "C": torch.nn.CrossEntropyLoss(),
                "R": torch.nn.CrossEntropyLoss(),
            }
        )
        self.datasets = {
            'train': OfficeHome(split='train'),
            'valid': OfficeHome(split='test')
        }

    def evaluate(self, model, loader):
        return OfficeHomeEvaluator.evaluate(model, loader, device=next(model.parameters()).device)

    def evaluate_task(self, model, loader):
        return OfficeHomeEvaluator.evaluate_task(model, loader, device=next(model.parameters()).device)

    def evaluate_each_task(self, model, loader):
        return OfficeHomeEvaluator.evaluate_each_task(model, loader, device=next(model.parameters()).device)


    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=32, test_batch=32, epochs=60, lr=1e-4, n_expert=4)
        return parser

    @staticmethod
    def get_model(args):
        if args.arch == 'lora_soft_router':
            return OfficeHomeModel_LoRASoftRouter(args)
        elif args.arch == 'lora_grad':
            return OfficeHomeModel_LoRAGrad(args)
        elif args.arch == 'Cross_stitch':
            return OfficeHomeModel_CrossStitch(args)
        elif args.arch == 'MMoE':
            return OfficeHomeModel_MMoE(args)
        elif args.arch == 'DSelect_k':
            return OfficeHomeModel_DSelect_k(args)
        elif args.arch == 'LTB':
            return OfficeHomeModel_LTB(args)
        elif args.arch == 'CGC':
            return OfficeHomeModel_CGC(args)
        elif args.arch == 'PLE':
            return OfficeHomeModel_PLE(args)
        elif args.arch == 'HPS':
            return OfficeHomeModel_HPS(args)
        elif args.arch == 'MTAN':
            return OfficeHomeModel_MTAN(args)
        else:
            return OfficeHomeModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.999))

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        

