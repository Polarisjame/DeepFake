import os
from src.video_swin_transformer import *
from torch.nn import BCELoss,BCEWithLogitsLoss,CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler, SGD, RMSprop
from src.utils import Logger, AverageMeter
from data.data_process import DeepFakeSet
from torch.nn.parallel import DataParallel 

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
        
        # GPU Parallel
        # device_ids = list(range(torch.cuda.device_count())) 
        # self.model = DataParallel(self.model, device_ids=device_ids).to(device_ids[0]) 
        
        self.log_step = args.log_step
        logger(getModelSize(model, logger))
        
        self.optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_decacy, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.train_epochs)
        self.lossF = CrossEntropyLoss()
        # self.optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.l2_decacy)
    
    def run_batch(self, video_feat, label):
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
                for iter_id, batch in enumerate(dataloader):
                    video_feat,label = batch
                    video_feat = video_feat.to(self.device)
                    label = label.to(self.device)
                    run_stats = self.run_batch(video_feat, label)
                    loss = run_stats['loss']
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()                    
                        if t % self.log_step == 0:
                            logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Train Loss {:3.5f} | Train Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                        loss, run_stats['acc']))
                    else:
                        logger('| epoch {:2d} | step {:4d} | lr {:.4E} | Val Loss {:3.5f} | Val Acc {:1.5f} '.format(epoch, t, lr,
                                                                                                        loss, run_stats['acc']))   
                    t+=1
                    loss_stat.update(loss)
                logger(f'Phase:{phase}, Avg Loss:{loss_stat.avg}')
                loss_stat.reset()
            if epoch % self.model_save == 0:
                torch.save(self.model.state_dict(), "./checkpoints/VST_deepfake_epoch{:d}.pth".format(epoch))

                    