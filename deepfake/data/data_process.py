import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.append('../')
from utils import *


class DeepFake(data.Dataset):

    def __init__(self, root, args, transforms=None, train=True):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        if train:
            self.dataset_path = os.path.join(root , 'trainset')
            label_path = os.path.join(root, 'train_label.txt')
        else:
            self.dataset_path = os.path.join(root , 'valset')
            label_path = os.path.join(root, 'val_label.txt')

        self.filepaths = [os.path.join(self.dataset_path, video) for video in os.listdir(self.dataset_path)]
        label_raw = pd.read_csv(label_path)
        raw_dict = {}
        for index in range(len(label_raw)):
            raw_dict[label_raw['video_name'][index]] = label_raw['target'][index]
        self.video_dict = raw_dict
        self.label_raw = label_raw
        self.num_frames = args.num_frames

    def __getitem__(self, index):
        """
        一次返回一段视频
        """
        file_root = self.filepaths[index]
        label = self.video_dict[file_root.split('/')[-1]]
        # print(img_path)
        video_feature_row = extract_frames(file_root, self.num_frames)
        video_feature = preprocess_frames(video_feature_row, 224)
        # if self.transform:
        #     video_feature = self.transform(video_feature)
        video_feature = torch.tensor(video_feature,dtype=torch.float32).view(3,-1,224,224)
        label_tensor = torch.zeros(2)
        label_tensor[label] = 1
        return video_feature, label

    def __len__(self):
        return len(self.filepaths)


class DeepFakeSet():
    def __init__(self, args, world_size=None, rank=None):
        super().__init__()
        # self.save_hyperparameters(args)
        self.valset = None
        self.trainset = None
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.args = args
        self.world_size = world_size
        self.rank = rank

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = DeepFake(root=self.args.data_root, train=True, args=self.args)
            # perm = torch.randperm(len(self.trainset))
            # self.trainset = self.trainset[perm]
            self.valset = DeepFake(root=self.args.data_root, train=False, args=self.args)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        if self.world_size is None:
            dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            sampler = DistributedSampler(self.trainset,
                                        num_replicas=self.world_size,
                                        rank=self.rank) 
            dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=False, sampler=sampler)
        # for data in dataloader:
        #     input, label = data
        #     print(label.data)
        return dataloader

    def val_dataloader(self):
        if self.world_size is None:
            dataloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        else:
            sampler = DistributedSampler(self.valset,
                                        num_replicas=self.world_size,
                                        rank=self.rank) 
            dataloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, sampler=sampler)

    def test_dataloader(self):
        return 0