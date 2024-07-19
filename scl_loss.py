import torch
import torch.nn as nn
import torch.nn.functional as F

def moe_cl_loss(fea, label, tau=1.):
        batch_size = fea.shape[0]
        fea = F.normalize(fea)
        sim = fea.mm(fea.t())  

        sim = (sim / tau).exp()
        label = label.unsqueeze(1).repeat(1, batch_size)
        loss = []
        sim = sim - sim.diag().diag()
        for i in range(batch_size):
            for j in range(batch_size):
                if label[j, i] == label[i, i]:
                    if j != i:
                        loss_ = -(sim[j, i] / sim[:, i].sum()).log()
                        loss.append(loss_)
        loss = torch.stack(loss).mean()
        return loss