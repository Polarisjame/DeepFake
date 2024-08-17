import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils import data
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from data.cuda_dataloader import CudaDataLoader
import sys
sys.path.append('../')
from src.utils import *
import threading


class DeepFake(data.Dataset):

    def __init__(self, root, args, event: threading.Event, transforms=None, train=True, logger=None, test=False):
        """
        主要目标： 获取所有图片的地址，并根据训练，验证，测试划分数据
        """
        if train:
            self.dataset_path = os.path.join(root , 'phase1', 'trainset')
            label_path = os.path.join(root, 'train_label.txt')
        elif not test:
            self.dataset_path = os.path.join(root , 'phase1', 'valset')
            label_path = os.path.join(root, 'val_label.txt')
        elif test:
            self.dataset_path = os.path.join(root , 'phase2', 'testset1seen')
            label_path = None
        
        self.train = train
        self.test = test
        self.filepaths = [os.path.join(self.dataset_path, video) for video in os.listdir(self.dataset_path)]
        if not test:
            label_raw = pd.read_csv(label_path)
            raw_dict = {}
            for index in range(len(label_raw)):
                raw_dict[label_raw['video_name'][index]] = label_raw['target'][index]
            self.video_dict = raw_dict
            self.label_raw = label_raw
        self.num_frames = args.num_frames
        self.modality = args.modality
        self.target_size = 224
        if test or not train:
            self.transform = T.Compose([
                T.Resize(self.target_size),
                T.CenterCrop(self.target_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
            ])
        else:
            self.transform = T.Compose([
                            T.Resize((self.target_size, self.target_size)),
                            T.RandomHorizontalFlip(),
                            T.RandomVerticalFlip(),
                            T.RandomRotation(90),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                            ])
        if self.modality == 'audio' or self.modality == 'fused':
            if train:
                extract_audio_img_path = os.path.join(root, 'trainAudioImgs')
            elif not test:
                extract_audio_img_path = os.path.join(root, 'ValAudioImgs')
            elif test:
                extract_audio_img_path = os.path.join(root, 'TestAudioImgs')
            if not os.path.exists(extract_audio_img_path) or args.force_generate:
                if not os.path.exists(extract_audio_img_path):
                    os.mkdir(extract_audio_img_path)
                logger("Processing Audio File!")

                for index, video_path in enumerate(self.filepaths):
                    target_dir = os.path.join(extract_audio_img_path,video_path.split('/')[-1][:-4] + '.jpg')
                    if os.path.exists(target_dir):
                        continue
                    if index % 100 == 0:
                        rate = int(index/len(self.filepaths)*100)
                        if train:
                            logger("Train:["+"*"*rate+"-"*(100-rate)+"]"+f" ({index}/{len(self.filepaths)})")
                        else:
                            logger("Val: ["+"*"*rate+"-"*(100-rate)+"]"+f" ({index}/{len(self.filepaths)})")
                    mel_spectrogram_image = generate_mel_spectrogram(video_path)
                    cv2.imwrite(target_dir, mel_spectrogram_image)
                logger("Processing Complete")
            else:
                logger("Audio File Has Previously Been Processed")
            self.audio_path = extract_audio_img_path
        # elif self.modality == 'paudio':
        #     if train:
        #         extract_audio_wav_path = os.path.join(root, 'trainAudioWav')
        #     else:
        #         extract_audio_wav_path = os.path.join(root, 'ValAudioWav')
                
        #     if not os.path.exists(extract_audio_wav_path) or args.force_generate:
        #         if not os.path.exists(extract_audio_wav_path):
        #             os.mkdir(extract_audio_wav_path)
        #     logger("Processing Pure Audio File!")
            
        #     thread_num = 8
        #     split_size = len(self.filepaths) // thread_num
        #     sub_lists = [self.filepaths[i:i+split_size] for i in range(0, len(self.filepaths), split_size)]
        #     remain_size = len(self.filepaths) - len(sub_lists)*split_size
        #     if remain_size > 0:
        #         sub_lists.append(self.filepaths[len(self.filepaths)-remain_size:])
        #         thread_num+=1
        #     threads = []
            
        #     for thread_index,sub_list in enumerate(sub_lists):
        #         t = threading.Thread(target=split_audio_wav_thread, args=(sub_list,extract_audio_wav_path,logger))
        #         threads.append(t)
        #         t.start()
        #         logger(f"Thread{thread_index} Start!")
        #     for thread_index,t in enumerate(threads):
        #         t.join()
        #         logger(f"Thread{thread_index} Finished!")
        #     for index, video_path in enumerate(self.filepaths):
        #         if index % 100 == 0:
        #             rate = int(index/len(self.filepaths)*100)
        #             if train:
        #                 logger("Train:["+"*"*rate+"-"*(100-rate)+"]"+f" ({index}/{len(self.filepaths)})")
        #             else:
        #                 logger("Val: ["+"*"*rate+"-"*(100-rate)+"]"+f" ({index}/{len(self.filepaths)})")
        #     self.paudio_path = extract_audio_wav_path

    def __getitem__(self, index):
        """
        一次返回一段视频
        """
        file_root = self.filepaths[index]
        if self.test:
            label_tensor=None
        else:
            label = self.video_dict[file_root.split('/')[-1]]
            label_tensor = torch.tensor(label,dtype=torch.float32)
        mask = None
        # print(img_path)
        if self.modality== 'video':
            feature = extract_frames(file_root, self.num_frames, self.target_size, self.transform)
        elif self.modality== 'audio':
            img_path = os.path.join(self.audio_path , file_root.split('/')[-1][:-4] + '.jpg')
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            feature = img
        elif self.modality== 'paudio':
                feature = extract_wav(file_root)
            # wav_path = os.path.join(self.paudio_path , file_root.split('/')[-1][:-4] + '.wav')
            #  y, sr = librosa.load(audio_path, sr=16000)
            # feature = y
            # feature = torch.tensor(y, dtype=torch.float32)
        elif self.modality=='fused':
            video_feat = extract_frames(file_root, self.num_frames, self.target_size, self.transform)
            audio_feat = self.transform(Image.open(os.path.join(self.audio_path , file_root.split('/')[-1][:-4] + '.jpg')).convert('RGB'))
            paudio_feat = extract_wav(file_root)
            feature = {"Video":video_feat, "Audio":audio_feat, "PAudio":paudio_feat}
        # label_tensor = torch.zeros(2)
        # label_tensor[label] = 1
        if self.test:
            label_tensor=None
            return feature, file_root.split('/')[-1]
        else:
            label = self.video_dict[file_root.split('/')[-1]]
            label_tensor = torch.tensor(label,dtype=torch.float32)
            return feature, label_tensor, file_root.split('/')[-1]

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
        self.modality = args.modality

    def setup(self, event:threading.Event, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = DeepFake(root=self.args.data_root, train=True, args=self.args, logger=self.logger, event=event)
            # perm = torch.randperm(len(self.trainset))
            # self.trainset = self.trainset[perm]
            self.valset = DeepFake(root=self.args.data_root, train=False, args=self.args, logger=self.logger, event=event)
            self.testset = DeepFake(root=self.args.data_root, train=False, test=True, args=self.args, logger=self.logger, event=event)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        if self.modality == 'paudio':
            dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_opt)
        elif self.modality == 'fused':
            dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=self.num_workers, collate_fn=fusion_collate)
            # dataloader = CudaDataLoader(dataloader, 'cuda')
        else:
            dataloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # for data in dataloader:
        #     input, label = data
        #     print(label.data)
        return dataloader

    def val_dataloader(self):
        if self.modality == 'paudio':
            dataloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_opt)
        elif self.modality == 'fused':
            dataloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=fusion_collate)
            # dataloader = CudaDataLoader(dataloader, 'cuda')
        else:
            dataloader = DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        if self.modality == 'paudio':
            dataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_opt)
        elif self.modality == 'fused':
            dataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=self.num_workers, collate_fn=fusion_collate)
            # dataloader = CudaDataLoader(dataloader, 'cuda')
        else:
            dataloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return dataloader