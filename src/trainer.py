import os
import torch
from torch.nn import BCELoss,BCEWithLogitsLoss,CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler, SGD, RMSprop
from src.utils import Logger, AverageMeter
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
        model.apply(weights_init)
        self.model = model
        self.model_save = args.model_save
        
        # GPU Parallel
        # device_ids = list(range(torch.cuda.device_count())) 
        # self.model = DataParallel(self.model, device_ids=device_ids).to(device_ids[0]) 
        
        self.log_step = args.log_step
        logger(getModelSize(model, logger))
        
        # self.optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_decacy, betas=(0.9, 0.999))
        if self.modality == 'video':
            self.optimizer = SGD([
                {"params":self.model.videoSwinT.parameters(),"lr":args.learning_rate},
                {"params":self.model.classsifier.parameters(),"lr":1e-3},], momentum=0.9, weight_decay=args.l2_decacy)
        else:
            self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs)
        self.lossF = CrossEntropyLoss()
    
    def run_batch(self, video_feat, label, masks):
        if self.modality == 'paudio':
            model_out = self.model(video_feat,masks)
        else:
            model_out = self.model(video_feat)
        loss = self.lossF(model_out, label)
        with torch.no_grad():
            _, indices = torch.max(model_out.to('cpu'), dim=1)
            # print(indices)
            correct = torch.sum(indices == label.to('cpu'))
            acc = correct.item() * 1.0 / len(label)
        return {
            'loss':loss,
            'acc':acc
        }
    
    def eval(self, dataloader, epoch, t, lr):
        logger = self.logger
        loss_stat = AverageMeter()
        with torch.no_grad():
            for iter_id, batch in enumerate(dataloader):
                video_feat,label = batch
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
                logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Val Loss {:3.5f} | Val Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                            loss, run_stats['acc']))
                loss_stat.update(loss) 
                t+=1  
            logger(f'Phase:val, Avg Loss:{loss_stat.avg}')
            loss_stat.reset()
        return t 
    def train(self, dataset:DeepFakeSet):
        trainloader = dataset.train_dataloader()
        valloader = dataset.val_dataloader()
        logger = self.logger
        
        loss_stat = AverageMeter()
        logger('[INFO] Start training, lr = {:.4f}'.format(self.optimizer.param_groups[0]['lr']))
        t=0
        for epoch in range(1, self.train_epochs):
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
                    self.eval(dataloader,epoch,t,lr)
                    continue
                for iter_id, batch in enumerate(dataloader):
                    if self.modality == 'paudio':
                        video_feat,label,masks = batch
                        masks=masks.to(self.device)
                    else:
                        video_feat,label = batch
                        masks = None
                    video_feat = video_feat.to(self.device)
                    label = label.to(self.device)
                    run_stats = self.run_batch(video_feat, label, masks)
                    loss = run_stats['loss']
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()                    
                    if t % self.log_step == 0:
                        logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Train Loss {:3.5f} | Train Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                    loss, run_stats['acc']))
                    t+=1
                    loss_stat.update(loss)
                logger(f'Phase:{phase}, Avg Loss:{loss_stat.avg}')
                loss_stat.reset()
            self.scheduler.step()
            if epoch % self.model_save == 0:
                torch.save(self.model.state_dict(), f"./checkpoints/VST_deepfake_modality{self.modality}_batch{self.batch_size}_epoch{epoch}.pth")

                    