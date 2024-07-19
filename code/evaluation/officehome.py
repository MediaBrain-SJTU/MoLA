import numpy as np
import torch
from tqdm import tqdm


class OfficeHomeEvaluator:

    @staticmethod
    def _compute_stats(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(4):
                if hrepr[task_label==th].shape[0] == 0:
                    continue

                if th == 0:
                    v_output = model.decoders["P"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc



    @staticmethod
    def _compute_stats_task(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            
            for th in range(4):
                hrepr = model.encoder(input.to(device), task_label=data[2].cuda())
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if isinstance(hrepr, list):
                    hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
                else:
                    hrepr = avgpool(hrepr).squeeze()
                if hrepr[task_label==th].shape[0] == 0:
                    continue
                if th == 0:
                    v_output = model.decoders["P"](hrepr[task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc



    @staticmethod
    def _compute_stats_each_task(model, data_loader, device):
        correct_list, total_list = torch.tensor([0]*4), torch.tensor([0]*4)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(4):
                if hrepr[th][task_label==th].shape[0] == 0:
                    continue
                if th == 0:
                    v_output = model.decoders["P"](hrepr[0][task_label==0])
                    correct_list[0] = correct_list[0] + torch.argmax(v_output, 1).eq(data[1].cuda()[task_label==0].squeeze()).float().sum().item()
                    total_list[0] = total_list[0] + v_output.shape[0]
                if th == 1:
                    l_output = model.decoders["A"](hrepr[1][task_label==1])
                    correct_list[1] = correct_list[1] + torch.argmax(l_output, 1).eq(data[1].cuda()[task_label==1].squeeze()).float().sum().item()
                    total_list[1] = total_list[1] + l_output.shape[0]
                if th == 2:
                    c_output = model.decoders["C"](hrepr[2][task_label==2])
                    correct_list[2] = correct_list[2] + torch.argmax(c_output, 1).eq(data[1].cuda()[task_label==2].squeeze()).float().sum().item()
                    total_list[2] = total_list[2] + c_output.shape[0]
                if th == 3:
                    s_output = model.decoders["R"](hrepr[3][task_label==3])
                    correct_list[3] = correct_list[3] + torch.argmax(s_output, 1).eq(data[1].cuda()[task_label==3].squeeze()).float().sum().item()
                    total_list[3] = total_list[3] + s_output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        v_acc = correct_list[0]/ total_list[0]
        l_acc = correct_list[1]/ total_list[1]
        c_acc = correct_list[2]/ total_list[2]
        s_acc = correct_list[3]/ total_list[3]

        return v_acc, l_acc, c_acc, s_acc




    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (
                p_acc,
                a_acc,
                c_acc,
                r_acc,
            ) = OfficeHomeEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (p_acc + a_acc + c_acc + r_acc)/4
        print(
            f"p_acc: {p_acc}, a_acc: {a_acc}, c_acc: {c_acc}, r_acc: {r_acc}, avg_acc: {avg_acc}; "
        )
        return {'p_acc': p_acc, 'a_acc': a_acc,
                'c_acc': c_acc, 'r_acc': r_acc, 'avg_acc': avg_acc}
