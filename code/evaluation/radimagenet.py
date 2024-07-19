import numpy as np
import torch
from tqdm import tqdm

class RadEvaluator:


    @staticmethod
    def _compute_stats(model, data_loader, device):


        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                hrepr = model.module.encoder(input.to(device))
            else:
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()

            for th in range(11):
                label_offset = tensor_num_classes_cumsum[th]
                if hrepr[task_label==th].shape[0] == 0:
                    continue
                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[task_label==th])
                
                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]

            pbar.update(1)
        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def _compute_stats_task(model, data_loader, device):

        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)

            for th in range(11):
                label_offset = tensor_num_classes_cumsum[th]

                if isinstance(model, torch.nn.DataParallel):
                    hrepr = model.module.encoder(input.to(device), task_label=data[2].cuda())
                else:
                    hrepr = model.encoder(input.to(device), task_label=data[2].cuda())
                    
                avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                if isinstance(hrepr, list):
                    hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
                else:
                    hrepr = avgpool(hrepr).squeeze()
                if hrepr[task_label==th].shape[0] == 0:
                    continue

                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[task_label==th])

                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]


            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def _compute_stats_each_task(model, data_loader, device):

        num_classes = [6,28,2,13,18,14,9,25,26,10,14]
        tensor_num_classes = torch.tensor(num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0).tolist()
        tensor_num_classes_cumsum = [0]+tensor_num_classes_cumsum[:-1]

        correct_list, total_list = torch.tensor([0]*11), torch.tensor([0]*11)

        pbar = tqdm(total=len(data_loader))
        for data in data_loader:
            input = data[0]
            task_label = torch.argmax(data[2], dim=1)
            if isinstance(model, torch.nn.DataParallel):
                hrepr = model.module.encoder(input.to(device))
            else:
                hrepr = model.encoder(input.to(device))
            avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            if isinstance(hrepr, list):
                hrepr = [avgpool(each_hrepr).squeeze() for each_hrepr in hrepr]
            else:
                hrepr = avgpool(hrepr).squeeze()
            for th in range(11):
                label_offset = tensor_num_classes_cumsum[th]
                if hrepr[th][task_label==th].shape[0] == 0:
                    continue

                if isinstance(model, torch.nn.DataParallel):
                    output = model.module.decoders[str(th)](hrepr[th][task_label==th])
                else:
                    output = model.decoders[str(th)](hrepr[th][task_label==th])

                correct_list[th] = correct_list[th] + torch.argmax(output, 1).eq((data[1].cuda()[task_label==th]-label_offset).squeeze()).float().sum().item()
                total_list[th] = total_list[th] + output.shape[0]

            pbar.update(1)

        pbar.clear()
        pbar.close()
        del pbar

        acc0 = correct_list[0]/ total_list[0]
        acc1 = correct_list[1]/ total_list[1]
        acc2 = correct_list[2]/ total_list[2]
        acc3 = correct_list[3]/ total_list[3]
        acc4 = correct_list[4]/ total_list[4]
        acc5 = correct_list[5]/ total_list[5]
        acc6 = correct_list[6]/ total_list[6]
        acc7 = correct_list[7]/ total_list[7]
        acc8 = correct_list[8]/ total_list[8]
        acc9 = correct_list[9]/ total_list[9]
        acc10 = correct_list[10]/ total_list[10]

        return acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10



    @staticmethod
    def evaluate(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats(model, data_loader, device)
        
        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}

    @staticmethod
    def evaluate_each_task(model, data_loader, device):
        model.eval()
        print(f"Compute statistics")
        with torch.no_grad():
            (acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10
            ) = RadEvaluator._compute_stats_each_task(model, data_loader, device)

        avg_acc = (acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7 + acc8 + acc9 + acc10)/11
        print(
            f"acc0: {acc0}, acc1: {acc1}, acc2: {acc2}, acc3: {acc3},   \
            acc4: {acc4}, acc5: {acc5}, acc6: {acc6}, acc7: {acc7},  \
            acc8: {acc8}, acc9: {acc9}, acc10: {acc10}, \
            avg_acc: {avg_acc}; "
        )
        return {'acc0': acc0, 'acc1': acc1, 'acc2': acc2, 'acc3': acc3, 'acc4': acc4, 'acc5': acc5,
            'acc6': acc6, 'acc7': acc7, 'acc8': acc8, 'acc9': acc9, 'acc10': acc10, 'avg_acc': avg_acc}