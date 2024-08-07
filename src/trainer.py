from collections import OrderedDict
import os
import numpy as np
import torch
from torch.nn import BCELoss,BCEWithLogitsLoss,CrossEntropyLoss,Sigmoid
from torch.optim import AdamW, lr_scheduler, SGD, RMSprop
from src.utils import Logger, AverageMeter, Drawer, plt
from data.data_process import DeepFakeSet
from torch.nn.parallel import DataParallel 
import torch.nn as nn
# from modelsize_estimate import modelsize
from gpu_mem_track import MemTracker
import time

def getModelSize(model,logger):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    logger('模型总大小为：{:.3f}MB'.format(all_size))
    return param_size, param_sum, buffer_size, buffer_sum, all_size

def weights_init(model):
    for m in model.modules():
        # 判断是否属于Conv2d
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight.data,gain=1.0)
            # 判断是否有偏置
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data,0.3)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Trainer():
    
    def __init__(self, model, args, device, dataset:DeepFakeSet, logger=None, processor=None) -> None:
        self.train_epochs = args.epochs
        self.device = device
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.modality = args.modality
        if logger is None:
            self.logger = Logger(f'./logs/deepFake_lr{self.lr}_batch{self.batch_size}.log')
        else:
            self.logger = logger
        if processor is not None:
            self.processor = processor
        model.cuda()
        self.model_s = model
        self.model_save = args.model_save
        self.trainloader = dataset.train_dataloader()
        self.valloader = dataset.val_dataloader()
        self.log_step = args.log_step
        self.start_epoch = 0
        self.accum_step = args.accum_step
        logger(getModelSize(self.model_s,logger))
        
        
        # GPU Parallel
        self.device_ids = list(range(torch.cuda.device_count())) 
        self.model = DataParallel(self.model_s, device_ids=self.device_ids).to(self.device_ids[0]) 
        
        
        # self.optimizer = AdamW([{'params': self.model.module.parameters(), 'initial_lr': args.learning_rate}], lr=args.learning_rate, weight_decay=args.l2_decacy, betas=(0.9, 0.999))
        if self.modality == 'video':
            self.optimizer = SGD([
                {"params":self.model.module.parameters(),
                 'initial_lr': args.learning_rate}],  lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
        else:
            self.optimizer = SGD(self.model.module.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs*int(len(self.trainloader)/self.accum_step))
        # self.scheduler = lr_scheduler.ExponentialLR(self.optimizer,0.98)
        # self.lossF = CrossEntropyLoss()
        self.lossF = BCELoss()
    
    def load_ckpt(self, args):
        logger = self.logger
        device = torch.device('cpu')
        if self.modality == 'fused':
            path_checkpoint = args.fused_ckpt_path 
            checkpoint = torch.load(path_checkpoint,map_location=device)
            state_dict = checkpoint['checkpoint']
            new_dict = OrderedDict()
            for k,v in state_dict.items():
                if not 'classify' in k:
                    new_dict[k] = v
            self.model_s.load_state_dict(new_dict, strict=False)
            torch.nn.init.constant_(self.model_s.classify.fc1.bias, 0.)
            torch.nn.init.constant_(self.model_s.classify.fc1.weight, 0.) 
            torch.nn.init.constant_(self.model_s.classify.fc2.bias, 0.)
            torch.nn.init.constant_(self.model_s.classify.fc2.weight, 0.) 
        else:
            if self.modality == 'audio':
                path_checkpoint = args.audio_ckpt_path   
            elif self.modality == 'video':
                path_checkpoint = args.vedio_ckpt_path 
            logger(f"Load Finetuned Model From:{path_checkpoint}")
            checkpoint = torch.load(path_checkpoint,map_location=device)
            state_dict = checkpoint['checkpoint']
            self.model_s.load_state_dict(state_dict, strict=False) 
        
        self.model = DataParallel(self.model_s, device_ids=self.device_ids).to(self.device_ids[0]) 
        # self.optimizer.load_state_dict(checkpoint['optimizer']) 
        self.start_epoch = checkpoint['epoch'] + 1  
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=(self.train_epochs-self.start_epoch+1)*int(len(self.trainloader)/self.accum_step))
        logger("Load Finetuned Model Succesfully")
        
    def run_batch(self, feature, label):
        start = time.time()
        model_out = self.model(feature)
        fini = time.time()
        print(f"Model Run : elapse {fini-start} secs")
        start = time.time()
        # print(model_out, label)
        loss = self.lossF(model_out, label)
        fini = time.time()
        print(f"Loss Cal : elapse {fini-start} secs")
        start = time.time()
        with torch.no_grad():
            # self.logger(f"{model_out}, {label}, {loss}")
            # _, indices = torch.max(model_out.to('cpu'), dim=1)
            # _, label_ind = torch.max(label.to('cpu'), dim=1)
            # print(indices)
            correct = torch.sum((model_out>=0.5)==label.to(int)).to('cpu')
            # correct = torch.sum(indices == label_ind.to('cpu'))
            acc = correct.numpy() * 1.0 / len(label)
        fini = time.time()
        print(f"Acc Cal : elapse {fini-start} secs")
        return {
            'loss':loss,
            'acc':acc
        }
    
    def submit(self,dataset):
        result_dict = {}
        logger = self.logger
        dataloader = dataset.val_dataloader()
        with torch.no_grad():
            for iter_id, batch in enumerate(dataloader):
                if self.modality == 'paudio':
                    audio_wav,label,filenames = batch
                    feature = self.processor(audio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values  # torch.Size([B, T])
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                elif self.modality == 'fused':
                    feat_dict,label,filenames = batch
                    label = label.to(self.device)
                    video_feat = feat_dict['Video'].to(self.device)
                    audio_feat = feat_dict['Audio'].to(self.device)
                    paudio_wav = feat_dict['PAudio']
                    paudio_feat = self.processor(paudio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values.to(self.device)
                    feature = (video_feat, audio_feat, paudio_feat)
                else:
                    feature,label,filenames = batch
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                model_out = self.model(feature)
                values = model_out.cpu()
                for ind,value in enumerate(values.numpy()):
                    filename = filenames[ind]
                    result_dict[filename] = value
                if iter_id % self.log_step == 0:
                    logger('|step {:4d} |'.format(iter_id))
        logger("Predict Done")
        print(result_dict)
        return result_dict
                
    def eval(self, dataloader, epoch, t, lr,val_loss_draw: Drawer):
        logger = self.logger
        loss_stat = AverageMeter()
        with torch.no_grad():
            for iter_id, batch in enumerate(dataloader):
                if self.modality == 'paudio':
                    audio_wav,label,_ = batch
                    feature = self.processor(audio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values  # torch.Size([B, T])
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                elif self.modality == 'fused':
                    feat_dict,label,_ = batch
                    label = label.to(self.device)
                    video_feat = feat_dict['Video'].to(self.device)
                    audio_feat = feat_dict['Audio'].to(self.device)
                    paudio_wav = feat_dict['PAudio']
                    paudio_feat = self.processor(paudio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values.to(self.device)
                    feature = (video_feat, audio_feat, paudio_feat)
                else:
                    feature,label,_ = batch
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                run_stats = self.run_batch(feature, label)
                loss = run_stats['loss']                    
                if t % self.log_step == 0:
                    logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Val Loss {:3.5f} | Val Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                                loss, run_stats['acc']))
                loss_stat.update(loss)
                if val_loss_draw is not None: 
                    val_loss_draw.update(loss)
                t+=1  
            logger(f'Phase:val, Avg Loss:{loss_stat.avg}')
            loss_stat.reset()
        return t 
    def train(self):
        trainloader = self.trainloader
        valloader = self.valloader
        logger = self.logger
        gpu_tracker = MemTracker(path='./gpu_track/')
        
        loss_stat = AverageMeter()
        train_loss_draw = Drawer(self.modality, 'train')
        val_loss_draw = Drawer(self.modality, 'val')
        logger('[INFO] Start training, lr = {:.6f}'.format(self.optimizer.param_groups[0]['lr']))
        t=0
        for epoch in range(self.start_epoch, self.train_epochs+1):
            lr = self.optimizer.param_groups[0]['lr']
            for phase in ['train', 'val']:
                if phase == 'train':
                    dataloader = trainloader
                    self.model.train()
                    self.model.to(self.device)
                else:
                    dataloader = valloader
                    self.model.eval()
                    self.model.to(self.device)
                    torch.cuda.empty_cache()
                    t = self.eval(dataloader,epoch,t,lr,val_loss_draw)
                    continue
                self.optimizer.zero_grad()
                start = time.time()
                for iter_id, batch in enumerate(dataloader):
                    print(f"---------------Iter: {iter_id}-------------")
                    fini = time.time()
                    print(f"Dataload : elapse {fini-start} secs")
                    start = time.time()
                    if self.modality == 'paudio':
                        audio_wav,label,_ = batch
                        feature = self.processor(audio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values.to(self.device)  # torch.Size([B, T])
                    elif self.modality == 'fused':
                        feat_dict,label,_ = batch
                        label = label.to(self.device)
                        video_feat = feat_dict['Video']
                        audio_feat = feat_dict['Audio']
                        paudio_wav = feat_dict['PAudio']
                        paudio_feat = self.processor(paudio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values.to(self.device)
                        feature = (video_feat, audio_feat, paudio_feat)
                    else:
                        feature,label,_ = batch
                        feature = feature.to(self.device)
                        label = label.to(self.device)
                    gpu_tracker.track()
                    fini = time.time()
                    print(f"Feat Prepare : elapse {fini-start} secs")
                    run_stats = self.run_batch(feature, label)

                    del feature,label
                    
                    start = time.time()
                    
                    gpu_tracker.track()
                    stage1 = torch.cuda.memory_allocated(0)/8/1024/1024
                    # logger('| epoch {:2d} | step {:4d} | Stage0 Mem Usage {} | Stage1 Mem Usage {}'.format(epoch, t, stage0, stage1))
                    loss = run_stats['loss']
                    loss_item = loss.item()
                    
                    if self.accum_step > 1:
                        loss = loss / self.accum_step
                    loss.backward()
                    
                    fini = time.time()
                    print(f"Backward : elapse {fini-start} secs")
                    if (iter_id+1) % self.accum_step == 0:
                        start = time.time()
                        t+=1
                        if t % self.log_step == 0:
                            lr = self.optimizer.param_groups[0]['lr']
                            # modelsize(self.model,feature, 4, logger)
                            logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Train Loss {:3.5f} | Train Acc {:1.5f} | MemUsage {:.4f}'.format(epoch, t, lr,
                                                                                                        loss_item, run_stats['acc'], stage1))
                            torch.cuda.empty_cache()
                        self.optimizer.step()  
                        self.scheduler.step()  
                        self.optimizer.zero_grad()
                        fini = time.time()
                        print(f"Optimizer Step : elapse {fini-start} secs")
                        
                    
                    start = time.time()
                    if (t+1) % self.model_save == 0:
                        path = f"./checkpoints/VST_deepfake_modality{self.modality}_batch{self.batch_size}_epoch{epoch}_step{t}.pth"
                        torch.save({
                            'epoch': epoch,
                            'checkpoint': self.model.module.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, path)
                        train_loss_draw.draw(epoch)
                        val_loss_draw.draw(epoch)
                    gpu_tracker.track()   
                    train_loss_draw.update(loss_item)
                    loss_stat.update(loss_item)
                    gpu_tracker.step()
                    fini = time.time()
                    print(f"Track: elapse {fini-start} secs")
                logger(f'Phase:{phase}, Avg Loss:{loss_stat.avg}')
            loss_stat.reset()
            train_loss_draw.reset()
            val_loss_draw.reset()
                    