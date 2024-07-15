import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
import sys
sys.path.append('../')
from src.utils import *
import threading


class DeepFake(data.Dataset):

    def __init__(self, root, args, transforms=None, train=True, logger=None):
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
        self.modality = args.modality
        if self.modality == 'audio':
            if train:
                extract_audio_img_path = os.path.join(root, 'trainAudioImgs')
            else:
                extract_audio_img_path = os.path.join(root, 'ValAudioImgs')
            if not os.path.exists(extract_audio_img_path) or args.force_generate:
                if not os.path.exists(extract_audio_img_path):
                    os.mkdir(extract_audio_img_path)
                logger("Processing Audio File!")
                # thread_num = 3
                # split_size = len(self.filepaths) // thread_num
                # sub_lists = [self.filepaths[i:i+split_size] for i in range(0, len(self.filepaths), split_size)]
                # remain_size = len(self.filepaths) - len(sub_lists)*split_size
                # if remain_size > 0:
                #     sub_lists.append(self.filepaths[len(self.filepaths)-remain_size:])
                #     thread_num+=1
                # threads = []
                
                # for thread_index,sub_list in enumerate(sub_lists):
                #     t = threading.Thread(target=process_list, args=(sub_list,extract_audio_img_path))
                #     threads.append(t)
                #     t.start()
                #     logger(f"Thread{thread_index} Start!")
                # for t in threads:
                #     t.join()
                #     logger(f"Thread{thread_index} Finished!")

                for video_path in self.filepaths:
                    mel_spectrogram_image = generate_mel_spectrogram(video_path)
                    cv2.imwrite(os.path.join(extract_audio_img_path,video_path.split('/')[-1][:-4] + '.jpg'), mel_spectrogram_image)
                logger("Processing Complete")
            else:
                logger("Audio File Has Previously Been Processed")
            self.audio_path = extract_audio_img_path
            self.transform = T.Compose([
                        T.Resize((224, 224)),
                        T.RandomHorizontalFlip(),
                        T.RandomVerticalFlip(),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        """
        一次返回一段视频
        """
        file_root = self.filepaths[index]
        label = self.video_dict[file_root.split('/')[-1]]
        # print(img_path)
        if self.modality== 'video':
            video_feature_row = extract_frames(file_root, self.num_frames)
            video_feature = preprocess_frames(video_feature_row, 224)
            video_feature = torch.tensor(video_feature,dtype=torch.float32).view(3,-1,224,224)
            feature = video_feature
        elif self.modality== 'audio':
            img_path = os.path.join(self.audio_path , file_root.split('/')[-1][:-4] + '.jpg')
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            feature = img
        # label_tensor = torch.zeros(2)
        # label_tensor[label] = 1
        return feature, label

    def __len__(self):
        return len(self.filepaths)


class DeepFakeSet():
    def __init__(self, args, world_size=None, rank=None, logger=None):
        super().__init__()
        # self.save_hyperparameters(args)
        self.valset = None
        self.trainset = None
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.logger = logger

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = DeepFake(root=self.args.data_root, train=True, args=self.args, logger=self.logger)
            # perm = torch.randperm(len(self.trainset))
            # self.trainset = self.trainset[perm]
            self.valset = DeepFake(root=self.args.data_root, train=False, args=self.args, logger=self.logger)
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
        return dataloader

    def test_dataloader(self):
        return 0