import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from abstrach_arch import AbsArchitecture


class HPS(AbsArchitecture):
    r"""Hard Parameter Sharing (HPS).

    This method is proposed in `Multitask Learning: A Knowledge-Based Source of Inductive Bias (ICML 1993) <https://dl.acm.org/doi/10.5555/3091529.3091535>`_ \
    and implemented by us. 
    """
    def __init__(self, task_name, encoder_class, rep_grad, multi_input, device, **kwargs):
        super(HPS, self).__init__(task_name, encoder_class, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
    
    def forward(self, inputs, task_name=None):
        r"""

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = []
        s_rep = self.encoder(inputs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out.append(ss_rep)
        return out
    