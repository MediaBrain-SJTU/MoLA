import torch
import torch.nn.functional as F
from . import mtl_benchmark


from code.models.nyu2 import (
    DepthDecoder,
    NormalDecoder,
    ResNet50Dilated,
    SemanticDecoder,
    ResNet50Dilated_LoraMix,
    ResNet50Dilated_LoraGrad,
    ResNet50Dilated_LoraRouter,
    ResNet50Dilated_LoraSoftRouter,
)
from code.models.segnet_mtan import MTANEncoder, MTANDepthDecoder, MTANNormalDecoder, MTANSemanticDecoder
from code.data.datasets import NYUv2
from code.evaluation.nyu2 import NYUv2Evaluator
from scl_loss import moe_cl_loss

from code.models.cross_stitch import Cross_stitch
from code.models.MMoE import MMoE
from code.models.DSelect_k import DSelect_k
from code.models.HPS import HPS
from code.models.CGC import CGC
from code.models.LTB import LTB
from code.models.PLE import PLE
from code.models.MTAN import MTAN


def semantic_loss(x_pred, target):
    # semantic loss: depth-wise cross entropy
    loss = F.nll_loss(x_pred, target.long(), ignore_index=-1)
    return loss


def depth_loss(x_pred, target):
    device = x_pred.device
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(device)
    # depth loss: l1 norm
    loss = torch.sum(torch.abs(x_pred - target) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


def normals_loss(x_pred, target):
    device = x_pred.device
    binary_mask = (torch.sum(target, dim=1) != 0).float().unsqueeze(1).to(device)
    # normal loss: dot product
    loss = 1 - torch.sum((x_pred * target) * binary_mask) / torch.nonzero(
        binary_mask, as_tuple=False
    ).size(0)
    return loss


class PSPNetNYUModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50Dilated(True)
        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = self.encoder.layer4


class PSPNetNYUModel_LoRAGrad(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet50Dilated_LoraGrad(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha, lora_layer=args.lora_layer, lora_rank=args.lora_rank)
        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = self.encoder.layer4


class PSPNetNYUModel_LoRASoftRouter(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()
        self.encoder = ResNet50Dilated_LoraSoftRouter(True, n_expert=args.n_expert, gamma=args.gamma, lora_alpha=args.lora_alpha, lora_layer=args.lora_layer, lora_rank=args.lora_rank)
        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = self.encoder.layer4


class PSPNetNYUModel_CrossStitch(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
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

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_MMoE(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 288, 384)
        kwargs['arch_args']['num_experts'] = [3]

        self.encoder = MMoE(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_DSelect_k(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 288, 384) 
        kwargs['arch_args']['num_experts'] = [3]
        kwargs['arch_args']['kgamma'] = 1.0 
        kwargs['arch_args']['num_nonzeros'] = 2

        self.encoder = DSelect_k(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_LTB(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
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

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_CGC(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 288, 384) 
        kwargs['arch_args']['num_experts'] = [1,1,1,1] 

        self.encoder = CGC(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_PLE(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
        rep_grad = False
        multi_input = False
        device = torch.device('cuda:0')
        kwargs = {}
        kwargs['arch_args'] = {}
        kwargs['arch_args']['img_size'] = (3, 288, 384) 
        kwargs['arch_args']['num_experts'] = [1,1,1,1] 

        self.encoder = PLE(task_name=task_name, 
                              encoder_class=encoder_class, 
                              rep_grad=rep_grad, 
                              multi_input=multi_input,
                              device=device,
                              kwargs=kwargs['arch_args'])

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_HPS(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
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

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None


class PSPNetNYUModel_MTAN(mtl_benchmark.MTLModel):
    def __init__(self, args):
        super().__init__()

        task_name = ['segmentation', 'normal', 'depth']
        def encoder_class(): 
            return ResNet50Dilated(True)
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

        self.decoders["SS"] = SemanticDecoder(num_class=13)
        self.decoders["NE"] = NormalDecoder()
        self.decoders["DE"] = DepthDecoder()
        self.last_shared_layer = None



class MTANNYUModel(mtl_benchmark.MTLModel):
    def __init__(self):
        super().__init__()
        self.encoder = MTANEncoder()
        for name, params in self.encoder.named_parameters():
            print(name, params.shape)
        self.decoders["SS"] = MTANSemanticDecoder(class_nb=13)
        self.decoders["NE"] = MTANNormalDecoder()
        self.decoders["DE"] = MTANDepthDecoder()
        self.last_shared_layer = None


@mtl_benchmark.register("nyuv2_pspnet")
class NYUBenchmark(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=["SS", "DE", "NE"],
            task_criteria={
                "SS": semantic_loss,
                "DE": depth_loss,
                "NE": normals_loss
            }
        )
        self.datasets = {
            'train': NYUv2(root=args.data_path, train=True, augmentation=True),
            'valid': NYUv2(root=args.data_path, train=False)
        }

    def evaluate(self, model, loader):
        return NYUv2Evaluator.evaluate(model, loader, device=next(model.parameters()).device)

    def evaluate_task(self, model, loader):
        return NYUv2Evaluator.evaluate_task(model, loader, device=next(model.parameters()).device)

    def evaluate_each_task(self, model, loader):
        return NYUv2Evaluator.evaluate_each_task(model, loader, device=next(model.parameters()).device)


    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=2, test_batch=16, epochs=200, lr=1e-4)
        parser.add_argument('--lr-decay-steps', type=int, default=100, help="Decrease LR every N epochs")
        parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='LR decay factor')
        return parser

    @staticmethod
    def get_model(args):
        if args.arch == 'lora_soft_router':
            return PSPNetNYUModel_LoRASoftRouter(args)
        elif args.arch == 'lora_grad':
            return PSPNetNYUModel_LoRAGrad(args)
        elif args.arch == 'Cross_stitch':
            return PSPNetNYUModel_CrossStitch(args)
        elif args.arch == 'MMoE':
            return PSPNetNYUModel_MMoE(args)
        elif args.arch == 'DSelect_k':
            return PSPNetNYUModel_DSelect_k(args)
        elif args.arch == 'LTB':
            return PSPNetNYUModel_LTB(args)
        elif args.arch == 'CGC':
            return PSPNetNYUModel_CGC(args)
        elif args.arch == 'PLE':
            return PSPNetNYUModel_PLE(args)
        elif args.arch == 'HPS':
            return PSPNetNYUModel_HPS(args)
        elif args.arch == 'MTAN':
            return PSPNetNYUModel_MTAN(args)
        else:
            return PSPNetNYUModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_factor)


@mtl_benchmark.register("nyuv2_mtan")
class NYUBenchmarkMTAN(mtl_benchmark.MTLBenchmark):
    def __init__(self, args):
        super().__init__(
            task_names=["SS", "DE", "NE"],
            task_criteria={
                "SS": semantic_loss,
                "DE": depth_loss,
                "NE": normals_loss
            }
        )
        self.datasets = {
            'train': NYUv2(root=args.data_path, train=True, augmentation=True),
            'valid': NYUv2(root=args.data_path, train=False)
        }

    def evaluate(self, model, loader):
        return NYUv2Evaluator.evaluate(model, loader, device=next(model.parameters()).device)

    @staticmethod
    def get_arg_parser(parser):
        parser.set_defaults(train_batch=2, test_batch=16, epochs=200, lr=1e-4)
        parser.add_argument('--lr-decay-steps', type=int, default=100, help="Decrease LR every N epochs")
        parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='LR decay factor')
        return parser

    @staticmethod
    def get_model(args):
        return MTANNYUModel()

    @staticmethod
    def get_optim(model, args):
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    @staticmethod
    def get_scheduler(optimizer, args):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_steps, gamma=args.lr_decay_factor)
