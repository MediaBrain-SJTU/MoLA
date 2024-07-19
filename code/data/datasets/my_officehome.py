import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
import random
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


officehome_name_dict = {
    'p': 'Product',
    'a': 'Art',
    'c': 'Clipart',
    'r': 'Real_World',
}

split_dict = {
    'train': 'train',
    'test': 'val',
}

class Officehome_MultiDomain(Dataset):
    def __init__(self, root_path='/mnt/data/medai/share/dataset/OfficeHome_compress', split='train', seed=0):
        
        self.root_path = root_path
        self.split = split
        assert self.split in ['train', 'test']
        self.split_file_list = []
        for domain_name in list(officehome_name_dict.values()):
            self.split_file_list.append(os.path.join(root_path, '{}_img_label_list.txt'.format(domain_name)))
        
        self.num_task = len(list(officehome_name_dict.values()))

        if self.split == 'train':
            self.transform = default_transform_train
        else:
            self.transform = default_transform_test
        self.seed = seed
                
        self.imgs = []
        self.labels = []
        self.domain_labels = []
        for each_split_file in self.split_file_list:
            with open(each_split_file, 'r') as f:
                txt_component = f.readlines()
            for line_txt in txt_component:
                line_txt = line_txt.replace('\n', '')
                line_txt_list = line_txt.split(' ')
                self.imgs.append(line_txt_list[0])
                self.labels.append(int(line_txt_list[1]))

                if 'Product' in each_split_file:
                    self.domain_labels.append(torch.tensor(0))
                if 'Art' in each_split_file:
                    self.domain_labels.append(torch.tensor(1))
                if 'Clipart' in each_split_file:
                    self.domain_labels.append(torch.tensor(2))
                if 'Real_World' in each_split_file:
                    self.domain_labels.append(torch.tensor(3))

        if self.split == 'train' or self.split == 'test':
            random.seed(self.seed)
            train_img, val_img = Officehome_MultiDomain.split_list(self.imgs, 0.9)
            random.seed(self.seed)
            train_label, val_label = Officehome_MultiDomain.split_list(self.labels, 0.9)
            random.seed(self.seed)
            train_domain_label, val_domain_label = Officehome_MultiDomain.split_list(self.domain_labels, 0.9)

        if self.split == 'train':
            self.imgs, self.labels, self.domain_labels = train_img, train_label, train_domain_label
        elif self.split == 'test':
            self.imgs, self.labels, self.domain_labels = val_img, val_label, val_domain_label

        print(len(self.imgs), np.unique((self.labels)))
        assert len(self.imgs) == len(self.labels) == len(self.domain_labels)

    @staticmethod
    def split_list(l, ratio):
        assert ratio > 0 and ratio < 1
        random.shuffle(l) # 打乱list
        train_size = int(len(l)*ratio)
        train_l = l[:train_size]
        val_l = l[train_size:]
        return train_l, val_l

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
        return 65
