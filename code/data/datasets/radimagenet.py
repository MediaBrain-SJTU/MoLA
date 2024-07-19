import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import io, transform
import torch
import torch.nn.functional as F


class RadData(Dataset):
    def __init__(self, base_path='/mnt/workspace/lihaolin/data', split="train", only_task=None):
        self.base_path = base_path
        self.split = split

        # image = (image-127.5)*2 / 255
        norm_params = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
        # norm_params = {'mean': [123.675, 116.28, 103.53],
        #                'std': [58.395, 57.12, 57.375]}
        normalize = transforms.Normalize(**norm_params)
        
        if split=="train":
            data_split_path = '/mnt/workspace/lihaolin/data/radiology_ai/label/rad_train_shuffled.txt'
            self.transform = transforms.Compose([
                # normalize,
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            data_split_path = '/mnt/workspace/lihaolin/data/radiology_ai/label/rad_test.txt'
            self.transform = transforms.Compose([
                # normalize,
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        self.images = []
        self.labels = []
        self.task_labels = []

        self.class_label_to_task_label_dict = {
            0: list(range(0,6)),
            1: list(range(6,34)),
            2: list(range(34,36)),
            3: list(range(36,49)),
            4: list(range(49,67)),
            5: list(range(67,81)),
            6: list(range(81,90)),
            7: list(range(90,115)),
            8: list(range(115,141)),
            9: list(range(141,151)),
            10: list(range(151,165)),
        }
        self.num_task = 11

        if only_task is not None:
            label_list = self.class_label_to_task_label_dict[only_task]
            min_lable = label_list[0]
            print(label_list, min_lable)

            with open(data_split_path, 'r') as f: 
                for line in f: 
                    path, label = line.strip().split(' ')
                    if int(label) in label_list:
                        self.images.append(os.path.join(self.base_path, path))
                        self.labels.append(int(label)-min_lable)
                        for each_key in self.class_label_to_task_label_dict.keys():
                            if int(label) in self.class_label_to_task_label_dict[each_key]:
                                self.task_labels.append(torch.tensor(each_key))

        else:
            # each_class_num = [0]*165
            # real_each_class_num = [0]*165
            # num_per_class = 10

            with open(data_split_path, 'r') as f: 
                for line in f: 
                    path, label = line.strip().split(' ')

                    # each_class_num[int(label)] = each_class_num[int(label)] + 1

                    # if each_class_num[int(label)] < num_per_class+1:

                    self.images.append(os.path.join(self.base_path, path))
                    self.labels.append(int(label))
                    for each_key in self.class_label_to_task_label_dict.keys():
                        if int(label) in self.class_label_to_task_label_dict[each_key]:
                            self.task_labels.append(torch.tensor(each_key))

                    # real_each_class_num[int(label)] = real_each_class_num[int(label)] + 1
        
        assert len(self.labels) == len(self.task_labels)
        # print(len(self.labels))

        # print(split, 'data size:', len(self.labels))
        # print(split, 'each_class_num:', each_class_num)
        # print(split, 'real_each_class_num:', real_each_class_num)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.images[index]).convert('RGB')

        # mean = np.array([123.675, 116.28, 103.53])
        # std = np.array([58.395, 57.12, 57.375])

        # x = np.array(x)
        # # print(x.shape, np.max(x), np.min(x))
        # x = (x-mean) / std
        # # x = (x-127.5)*2 / 255
        # x = (x-np.min(x))/(np.max(x)-np.min(x))*255
        # # print(x.shape, np.max(x), np.min(x))
        # x = Image.fromarray(x.astype(np.uint8))

        y = self.labels[index]
        task_y = F.one_hot(self.task_labels[index], num_classes=self.num_task) 
        # alphas = torch.eye(num_task)[:num_task][task_label]

        if isinstance(self.transform,list):
            sample1 = self.transform[0](x)
            sample2 = self.transform[1](x)
            return [sample1, sample2], y, task_y

        else:
            x = self.transform(x)
            return x, y, task_y


    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.images)



