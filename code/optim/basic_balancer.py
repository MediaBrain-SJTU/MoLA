from collections import defaultdict
import torch
import numpy as np
from . import mtl_metrics
from scl_loss import moe_cl_loss

class BasicBalancer(torch.nn.Module):
    def __init__(self, compute_stats=False):
        super().__init__()
        self.compute_stats = compute_stats
        self.info = None
        self.losses = defaultdict(float)

    def set_losses(self, losses):
        self.losses = {task_id: float(losses[task_id]) for task_id in losses}

    def compute_metrics(self, G: torch.Tensor):
        self.info = mtl_metrics.compute_metrics(G)

    def add_model_parameters(self, model):
        pass

    @staticmethod
    def zero_grad_model(model):
        model.zero_grad()

    @staticmethod
    def apply_decoder_scaling(decoders, weights):
        for i, decoder in enumerate(decoders.values()):
            for p in decoder.parameters():
                if p.grad is not None:
                    p.grad.mul_(weights[i])

    @staticmethod
    def scale_task_specific_params(task_specific_params: dict, weights: dict):
        for task_id in task_specific_params:
            for p in task_specific_params[task_id]:
                if p.grad is not None:
                    p.grad.mul_(weights[task_id])

    @staticmethod
    def set_encoder_grad(encoder, grad_vec):
        offset = 0
        for p in encoder.parameters():
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset

    @staticmethod
    def set_shared_grad(shared_params, grad_vec):
        offset = 0
        for p in shared_params:
            if p.grad is None:
                continue
            _offset = offset + p.grad.shape.numel()
            p.grad.data = grad_vec[offset:_offset].view_as(p.grad)
            offset = _offset

    @staticmethod
    def get_G_wrt_shared(losses, shared_params, update_decoder_grads=False):
        grads = []
        for task_id in losses:
            cur_loss = losses[task_id]
            if not update_decoder_grads:
                grad = torch.cat([p.flatten() if p is not None else torch.zeros_like(shared_params[i]).flatten()
                                  for i, p in enumerate(torch.autograd.grad(cur_loss, shared_params,
                                                               retain_graph=True, allow_unused=True))])
            else:
                for p in shared_params:
                    if p.grad is not None:
                        p.grad.data.zero_()

                cur_loss.backward(retain_graph=True)
                grad = torch.cat([p.grad.flatten().clone() if p.grad is not None else torch.zeros_like(p).flatten()
                                  for p in shared_params])

            grads.append(grad)

        for p in shared_params:
            if p.grad is not None:
                p.grad.data.zero_()

        return torch.stack(grads, dim=0)

    @staticmethod
    def get_model_G_wrt_shared(hrepr, targets, encoder, decoders, criteria, loss_fn=None,
                               update_decoder_grads=False, return_losses=False):
        if loss_fn is None:
            loss_fn = lambda task_task_id: criteria[task_task_id](decoders[task_task_id](hrepr), targets[task_task_id])

        grads = []
        losses = {}
        for task_id in criteria:
            cur_loss = loss_fn(task_id)
            if not update_decoder_grads:
                grad = torch.cat([p.flatten()
                                  for p in torch.autograd.grad(cur_loss, encoder.parameters(),
                                                               retain_graph=True, allow_unused=True)
                                  if p is not None])
            else:
                encoder.zero_grad()
                cur_loss.backward(retain_graph=True)
                grad = torch.cat([p.grad.flatten().clone() for p in encoder.parameters() if p.grad is not None])

            grads.append(grad)
            losses[task_id] = cur_loss

        grads = torch.stack(grads, dim=0)
        if return_losses:
            return grads, losses
        else:
            return grads

    @staticmethod
    def get_model_G_wrt_hrepr(hrepr, targets, model, criteria, loss_fn=None,
                              update_decoder_grads=False, return_losses=False):

        _hrepr = hrepr.data.detach().clone().requires_grad_(True)
        if loss_fn is None:
            loss_fn = lambda task_task_id: criteria[task_task_id](model.decoders[task_task_id](_hrepr),
                                                                  targets[task_task_id])

        grads = []
        losses = {}
        for task_id in criteria:
            cur_loss = loss_fn(task_id)
            if not update_decoder_grads:
                grad = torch.cat([p.flatten()
                                  for p in torch.autograd.grad(cur_loss, _hrepr,
                                                               retain_graph=False, allow_unused=True)
                                  if p is not None])
            else:
                if _hrepr.grad is not None:
                    _hrepr.grad.data.zero_()
                cur_loss.backward(retain_graph=False)
                grad = _hrepr.grad.flatten().clone()

            grads.append(grad)
            losses[task_id] = cur_loss

        grads = torch.stack(grads, dim=0)
        if return_losses:
            return grads, losses
        else:
            return grads

    @staticmethod
    def compute_losses(data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)


        if isinstance(model, torch.nn.DataParallel):
            hrepr = model.module.encoder(data)
        else:
            hrepr = model.encoder(data)

        losses = {}
        task_th = 0
        for task_id in criteria:
            if not isinstance(hrepr, list):
                if isinstance(model, torch.nn.DataParallel):
                    losses[task_id] = criteria[task_id](model.module.decoders[task_id](hrepr), targets[task_id])
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr), targets[task_id])
            else:
                losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_th]), targets[task_id])
                task_th = task_th + 1

        return losses, hrepr

    @staticmethod
    def compute_losses_task(data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)
        losses = {}
        task_th = 0
        for task_id in criteria:
            hrepr = model.encoder(data, task_label=task_th)
            losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr), targets[task_id])
            task_th = task_th + 1
        return losses, hrepr


    def step_with_model(self, data: torch.Tensor, targets: dict, model: torch.nn.Module, criteria: dict, args,
                        **kwargs) -> None:

        if args.benchmark in ['vlcs_resnet50', 'officehome_resnet50', 'vlcs_understanding']:
            if 'grad' in args.arch:
                losses, hrepr = self.compute_losses_task_multidomain(data=data, targets=targets, task_targets=kwargs['task_targets'], model=model, criteria=criteria)
            elif 'soft_router' in args.arch:
                losses, hrepr = self.compute_losses_soft_multidomain(data=data, targets=targets, task_targets=kwargs['task_targets'], model=model, criteria=criteria)
            else:
                losses, hrepr = self.compute_losses_multidomain(data=data, targets=targets, task_targets=kwargs['task_targets'], model=model, criteria=criteria)
        elif args.benchmark in ['radimagenet_resnet18', 'medmnist_resnet18']:
            if args.benchmark == 'radimagenet_resnet18':
                num_task = 11
            if args.benchmark == 'medmnist_resnet18':
                num_task = 9
            if 'grad' in args.arch:
                losses, hrepr = self.compute_losses_task_tasklabel(data=data, targets=targets, task_targets=kwargs['task_targets'], model=model, criteria=criteria, num_task=num_task)
            else:
                losses, hrepr = self.compute_losses_tasklabel(data=data, targets=targets, task_targets=kwargs['task_targets'], model=model, criteria=criteria, num_task=num_task)
        else:
            if 'grad' in args.arch:
                losses, hrepr = self.compute_losses_task(data, targets, model, criteria)
            else:
                losses, hrepr = self.compute_losses(data, targets, model, criteria)


        if isinstance(model, torch.nn.DataParallel):
            self.step(losses=losses,
                    shared_params=list(model.module.encoder.parameters()),
                    task_specific_params={task_id: list(model.module.decoders.parameters()) for task_id in model.module.decoders},
                    shared_representation=hrepr,
                    last_shared_layer_params=list(model.module.last_shared_layer.parameters())
                                            if model.module.last_shared_layer is not None
                                            else None)
        else:
            self.step(losses=losses,
                    shared_params=list(model.encoder.parameters()),
                    task_specific_params={task_id: list(model.decoders.parameters()) for task_id in model.decoders},
                    shared_representation=hrepr,
                    last_shared_layer_params=list(model.last_shared_layer.parameters())
                                            if model.last_shared_layer is not None
                                            else None)




    @staticmethod
    def compute_losses_tasklabel(data: torch.Tensor, targets: dict, task_targets, model: torch.nn.Module, criteria: dict, num_task, **kwargs):

        if num_task == 11:
            num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        if num_task == 9:
            num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2] 
            targets = targets.squeeze()
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist() # tensor([  6,  34,  36,  49,  67,  81,  90, 115, 141, 151, 165])
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        BasicBalancer.zero_grad_model(model)

        task_label = torch.argmax(task_targets, dim=1)

        if isinstance(model, torch.nn.DataParallel):
            hrepr = model.module.encoder(data)
        else:
            hrepr = model.encoder(data)
        
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if isinstance(hrepr, list):
            hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
        else:
            hrepr = avgpool(hrepr).squeeze()

        losses = {}
        task_th = 0
        for task_id in criteria:
            label_offset = tensor_num_classes_cumsum[task_th]
            if not isinstance(hrepr, list):
                if hrepr[task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    if isinstance(model, torch.nn.DataParallel):
                        losses[task_id] = criteria[task_id](model.module.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th]-label_offset)
                    else:
                        sample_num = hrepr[task_label==task_th].shape[0]
                        losses[task_id] = (sample_num/hrepr.shape[0]) * criteria[task_id](model.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th]-label_offset)
                    task_th = task_th + 1
            else:
                if hrepr[task_th][task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    if isinstance(model, torch.nn.DataParallel):
                        losses[task_id] = criteria[task_id](model.module.decoders[task_id](hrepr[task_th][task_label==task_th]), targets[task_label==task_th]-label_offset)
                    else:
                        losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_th][task_label==task_th]), targets[task_label==task_th]-label_offset)
                    task_th = task_th + 1

        return losses, hrepr

    @staticmethod
    def compute_losses_task_tasklabel(data: torch.Tensor, targets: dict, task_targets, model: torch.nn.Module, criteria: dict, num_task, **kwargs):
        if num_task == 11:
            num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        if num_task == 9:
            num_classes = [9, 4, 11, 8, 2, 7, 11, 11, 2] 
            targets = targets.squeeze()
            
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist() # tensor([  6,  34,  36,  49,  67,  81,  90, 115, 141, 151, 165])
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        BasicBalancer.zero_grad_model(model)
        task_label = torch.argmax(task_targets, dim=1)
        losses = {}
        task_th = 0
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        for task_id in criteria:
            label_offset = tensor_num_classes_cumsum[task_th]

            if isinstance(model, torch.nn.DataParallel):
                hrepr = model.module.encoder(data, task_label=task_targets)
            else:
                hrepr = model.encoder(data, task_label=task_targets)

            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()

            if hrepr[task_label==task_th].shape[0] == 0:
                task_th = task_th + 1
                continue
            else:
                if isinstance(model, torch.nn.DataParallel):
                    losses[task_id] = criteria[task_id](model.module.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th]-label_offset)
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th]-label_offset)
                task_th = task_th + 1

        return losses, hrepr




    @staticmethod
    def compute_losses_multidomain(data: torch.Tensor, targets: dict, task_targets, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)

        task_label = torch.argmax(task_targets, dim=1)

        if isinstance(model, torch.nn.DataParallel):
            hrepr = model.module.encoder(data)
        else:
            hrepr = model.encoder(data)
        
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if isinstance(hrepr, list):
            hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            for th in range(len(hrepr)):
                each_hrepr = hrepr[th]
                if len(each_hrepr.shape) == 1:
                    each_hrepr = each_hrepr.unsqueeze(0)
                    hrepr[th] = each_hrepr
        else:
            hrepr = avgpool(hrepr).squeeze()
            if len(hrepr.shape) == 1:
                hrepr = hrepr.unsqueeze(0)

        losses = {}
        task_th = 0
        for task_id in criteria:
            if not isinstance(hrepr, list):
                if hrepr[task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th])
                    task_th = task_th + 1
            else:
                if hrepr[task_th][task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_th][task_label==task_th]), targets[task_label==task_th])
                    task_th = task_th + 1

        return losses, hrepr




    @staticmethod
    def compute_losses_soft_multidomain(data: torch.Tensor, targets: dict, task_targets, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)

        task_label = torch.argmax(task_targets, dim=1)

        if isinstance(model, torch.nn.DataParallel):
            hrepr, scores = model.module.encoder(data, return_scores=True)
        else:
            hrepr, scores = model.encoder(data, return_scores=True)
        
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if isinstance(hrepr, list):
            hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            for th in range(len(hrepr)):
                each_hrepr = hrepr[th]
                if len(each_hrepr.shape) == 1:
                    each_hrepr = each_hrepr.unsqueeze(0)
                    hrepr[th] = each_hrepr
        else:
            hrepr = avgpool(hrepr).squeeze()
            if len(hrepr.shape) == 1:
                hrepr = hrepr.unsqueeze(0)

        losses = {}
        task_th = 0
        for task_id in criteria:
            if not isinstance(hrepr, list):
                if hrepr[task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th])
                    task_th = task_th + 1
            else:
                if hrepr[task_th][task_label==task_th].shape[0] == 0:
                    task_th = task_th + 1
                    continue
                else:
                    losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_th][task_label==task_th]), targets[task_label==task_th])
                    task_th = task_th + 1
        losses['scl'] = 0.1*moe_cl_loss(scores, task_label)

        return losses, hrepr



    @staticmethod
    def compute_losses_task_multidomain(data: torch.Tensor, targets: dict, task_targets, model: torch.nn.Module, criteria: dict, **kwargs):
        BasicBalancer.zero_grad_model(model)
        task_label = torch.argmax(task_targets, dim=1)
        losses = {}
        task_th = 0
        avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        for task_id in criteria:
            hrepr = model.encoder(data, task_label=task_targets)

            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()

            if hrepr[task_label==task_th].shape[0] == 0:
                task_th = task_th + 1
                continue
            else:
                losses[task_id] = criteria[task_id](model.decoders[task_id](hrepr[task_label==task_th]), targets[task_label==task_th])
                task_th = task_th + 1

        return losses, hrepr


    def step(self, losses, shared_params, task_specific_params, shared_representation=None,
             last_shared_layer_params=None) -> None:
        raise NotImplementedError("Balancer requires model to be specified. "
                                  "Use 'step_with_model' method for this balancer")
