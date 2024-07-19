# base_path = '/mnt/data/medai/share/dataset/VLCS'

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")
    
default_transform_train = transforms.Compose(
            [transforms.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),            
            ])

default_transform_test = transforms.Compose([
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

vlcs_name_dict = {
    'v': 'PASCAL',
    'l': 'LABELME',
    'c': 'CALTECH',
    's': 'SUN',
}


split_dict = {
    'train': 'train',
    'test': 'crossval',
    # 'total': 'full',
    # 'total': 'test',
}

class_dict = {
    '0': 'bird',
    '1': 'car',
    '2': 'chair',
    '3': 'dog',
    '4': 'person'
}

class VLCS_MultiDomain(Dataset):
    def __init__(self, root_path='/mnt/data/medai/share/dataset/VLCS', split='train'):
        
        self.root_path = root_path
        self.split = split
        assert self.split in ['train', 'test']
        self.split_file_list = []
        for domain_name in list(vlcs_name_dict.values()):
            self.split_file_list.append(os.path.join(root_path, f'{domain_name}_{split_dict[self.split]}' + '.txt'))
        
        self.num_task = len(list(vlcs_name_dict.values()))

        if self.split == 'train':
            self.transform = default_transform_train
        else:
            self.transform = default_transform_test
                
        self.imgs = []
        self.labels = []
        self.domain_labels = []
        for each_split_file in self.split_file_list:
            with open(each_split_file, 'r') as f:
                txt_component = f.readlines()
            for line_txt in txt_component:
                line_txt = line_txt.replace('\n', '')
                line_txt = line_txt.split(' ')
                self.imgs.append(os.path.join(root_path, line_txt[0]))
                self.labels.append(int(line_txt[1]))

                if 'PASCAL' in each_split_file:
                    self.domain_labels.append(torch.tensor(0))
                if 'LABELME' in each_split_file:
                    self.domain_labels.append(torch.tensor(1))
                if 'CALTECH' in each_split_file:
                    self.domain_labels.append(torch.tensor(2))
                if 'SUN' in each_split_file:
                    self.domain_labels.append(torch.tensor(3))

        print(len(self.imgs), np.unique(self.labels), np.unique(self.domain_labels))
        assert len(self.imgs) == len(self.labels) == len(self.domain_labels)

    def __getitem__(self, index):
        """Generates one sample of data"""
        x = Image.open(self.imgs[index]).convert('RGB')
        y = self.labels[index]
        task_y = F.one_hot(self.domain_labels[index], num_classes=self.num_task) 
        # alphas = torch.eye(num_task)[:num_task][task_label]

        if len(x.split()) != 3:
            x = transforms.Grayscale(num_output_channels=3)(x)

        if self.transform is not None:
            x = self.transform(x)
        return x, y, task_y


    def __len__(self):
        return len(self.imgs)


    @property
    def task_num(self):
        return self.num_task

    @property
    def class_num(self):
        return 5

