import os
import numpy as np
import torch
from torch.nn import BCELoss,BCEWithLogitsLoss,CrossEntropyLoss,Sigmoid
from torch.optim import AdamW, lr_scheduler, SGD, RMSprop
from src.utils import Logger, AverageMeter, Drawer, plt
from data.data_process import DeepFakeSet
from torch.nn.parallel import DataParallel 
import torch.nn as nn

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
    
    def __init__(self, model, args, device, logger=None) -> None:
        self.train_epochs = args.epochs
        self.device = device
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.modality = args.modality
        if logger is None:
            self.logger = Logger(f'./logs/deepFake_lr{self.lr}_batch{self.batch_size}.log')
        else:
            self.logger = logger
        model.cuda()
        self.model = model
        self.model_save = args.model_save
        self.proj_to_class = Sigmoid()
        
        # GPU Parallel
        device_ids = list(range(torch.cuda.device_count())) 
        self.model = DataParallel(self.model, device_ids=device_ids).to(device_ids[0]) 
        
        self.log_step = args.log_step
        self.start_epoch = 0
        logger(getModelSize(model, logger))
        
        # self.optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_decacy, betas=(0.9, 0.999))
        if self.modality == 'video':
            self.optimizer = SGD([
                {"params":self.model.module.parameters(),
                 'initial_lr': args.learning_rate}],  lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
        else:
            self.optimizer = SGD([{'params': self.model.module.parameters(), 'initial_lr': args.learning_rate}], lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs, last_epoch=self.start_epoch-1)
        # self.lossF = CrossEntropyLoss()
        self.lossF = BCELoss()
    
    def load_ckpt(self, args):
        logger = self.logger
        device = torch.device('cpu')
        if self.modality == 'audio':
            if args.audio_ckpt_path is not None:
                path_checkpoint = args.audio_ckpt_path  
                logger(f"Load Finetuned Model From:{path_checkpoint}")
                checkpoint = torch.load(path_checkpoint,map_location=device)  
                self.model.load_state_dict(checkpoint['checkpoint']) 
                self.optimizer.load_state_dict(checkpoint['optimizer']) 
                self.start_epoch = checkpoint['epoch'] + 1  
        elif self.modality == 'video':
            if args.vedio_ckpt_path is not None:
                path_checkpoint = args.vedio_ckpt_path  
                logger(f"Load Finetuned Model From:{path_checkpoint}")
                checkpoint = torch.load(path_checkpoint,map_location=device)  
                self.model.load_state_dict(checkpoint['model']) 
                self.optimizer.load_state_dict(checkpoint['optimizer']) 
                self.start_epoch = checkpoint['epoch'] + 1  
        logger("Load Finetuned Model Succesfully")
        
    def run_batch(self, video_feat, label, masks):
        if self.modality == 'paudio':
            model_out = self.model(video_feat,masks)
        else:
            model_out, _ = self.model(video_feat)
        loss = self.lossF(model_out, label)
        with torch.no_grad():
            # self.logger(f"{model_out}, {label}, {loss}")
            # _, indices = torch.max(model_out.to('cpu'), dim=1)
            # print(indices)
            correct = torch.sum((model_out>=0.5)==label.to(int)).to('cpu')
            # correct = torch.sum(indices == label.to('cpu'))
            acc = correct.numpy() * 1.0 / len(label)
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
                video_feat,label,filenames = batch
                video_feat = video_feat.to(self.device)
                label = label.to(self.device)
                if self.modality == 'paudio':
                    feature = video_feat[:,:,:-1]
                    mask = video_feat[:,:,-1]
                else:
                    feature = video_feat
                    mask = None
                if self.modality == 'paudio':
                    model_out = self.model(video_feat,mask)
                else:
                    model_out,_ = self.model(video_feat)
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
                video_feat,label,_ = batch
                video_feat = video_feat.to(self.device)
                label = label.to(self.device)
                if self.modality == 'paudio':
                    feature = video_feat[:,:,:-1]
                    mask = video_feat[:,:,-1]
                else:
                    feature = video_feat
                    mask = None
                run_stats = self.run_batch(feature, label, mask)
                loss = run_stats['loss']                    
                if t % self.log_step == 0:
                    logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Val Loss {:3.5f} | Val Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                                loss, run_stats['acc']))
                loss_stat.update(loss) 
                val_loss_draw.update(loss)
                t+=1  
            logger(f'Phase:val, Avg Loss:{loss_stat.avg}')
            loss_stat.reset()
        return t 
    def train(self, dataset:DeepFakeSet):
        trainloader = dataset.train_dataloader()
        valloader = dataset.val_dataloader()
        logger = self.logger
        
        loss_stat = AverageMeter()
        train_loss_draw = Drawer(self.modality, 'train')
        val_loss_draw = Drawer(self.modality, 'val')
        logger('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        t=0
        for epoch in range(self.start_epoch+1, self.train_epochs+1):
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
                for iter_id, batch in enumerate(dataloader):
                    if self.modality == 'paudio':
                        video_feat,label,_ = batch
                        feature = video_feat[:,:,:-1]
                        mask = video_feat[:,:,-1]
                        mask=mask.to(self.device)
                    else:
                        feature,label,_ = batch
                        masks = None
                    feature = feature.to(self.device)
                    label = label.to(self.device)
                    run_stats = self.run_batch(feature, label, masks)
                    loss = run_stats['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()                    
                    if t % self.log_step == 0:
                        logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Train Loss {:3.5f} | Train Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                    loss, run_stats['acc']))
                    train_loss_draw.update(loss)
                    t+=1
                    loss_stat.update(loss)
                logger(f'Phase:{phase}, Avg Loss:{loss_stat.avg}')
                loss_stat.reset()
            self.scheduler.step()
            if epoch % self.model_save == 0:
                path = f"./checkpoints/VST_deepfake_modality{self.modality}_batch{self.batch_size}_epoch{epoch}.pth"
                torch.save({
                    'epoch': epoch,
                    'checkpoint': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, path)
                train_loss_draw.draw(epoch)
                val_loss_draw.draw(epoch)
                    