import torch
from .. import basic_balancer
from .. import balancers

@balancers.register("ourbase")
class OurBaseBalancer(basic_balancer.BasicBalancer):


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


        total_loss = sum(losses.values())
        total_loss.backward()
        self.set_losses(losses)

