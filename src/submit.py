from collections import OrderedDict
import torch
from src.utils import Logger, GpuInfoTracker, load_pretrained
from gpu_mem_track import MemTracker
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

class SubmitCtl():
    def __init__(self, model, args, device, dataset:DeepFakeSet, logger=None, processor=None):
        self.device = device
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
        self.testloader = dataset.test_dataloader()
        self.log_step = args.log_step
        self.gpu_log = GpuInfoTracker()
        logger(getModelSize(self.model_s,logger))
        
        # GPU Parallel
        self.device_ids = list(range(torch.cuda.device_count())) 
        self.model = DataParallel(self.model_s, device_ids=self.device_ids).to(self.device_ids[0]) 
    
    def load_ckpt(self, args):
        logger = self.logger
        device = torch.device('cpu')
        if self.modality == 'fused':
            path_checkpoint = args.fused_ckpt_path 
            checkpoint = torch.load(path_checkpoint,map_location=device)
            state_dict = checkpoint['checkpoint']
            # new_dict = OrderedDict()
            # for k,v in state_dict.items():
            #     if not 'classify' in k:
            #         new_dict[k] = v
            self.model_s.load_state_dict(state_dict, strict=False)
            # GPU Parallel
            # torch.nn.init.constant_(self.model_s.classify.fc2.bias, 0.)
            # torch.nn.init.xavier_uniform_(self.model_s.classify.fc2.weight) 
        else:
            if self.modality == 'audio':
                path_checkpoint = args.audio_ckpt_path
                load_pretrained(args,self.model_s,self.logger)   
            elif self.modality == 'video':
                path_checkpoint = args.vedio_ckpt_path 
                checkpoint = torch.load(path_checkpoint,map_location=device)
                state_dict = checkpoint['checkpoint']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.model_s.load_state_dict(new_state_dict, strict=False) 
        logger(f"Load Finetuned Model From:{path_checkpoint}")
    
        self.device_ids = list(range(torch.cuda.device_count())) 
        self.model = DataParallel(self.model_s, device_ids=self.device_ids).to(self.device_ids[0]) 
    
    def submit(self):
        result_dict = {}
        logger = self.logger
        track = GpuInfoTracker(100)
        gpu_tracker = MemTracker(path='./gpu_track/')
        dataloader = self.testloader
        fileName='prediction.csv'
        self.model.eval()
        with open(fileName, 'a') as f:
            with torch.no_grad():
                for iter_id, batch in enumerate(dataloader):
                    if self.modality == 'paudio':
                        audio_wav,filenames = batch
                        feature = self.processor(audio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values  # torch.Size([B, T])
                        feature = feature.to(self.device)
                    elif self.modality == 'fused':
                        feat_dict,filenames = batch
                        video_feat = feat_dict['Video'].to(self.device)
                        audio_feat = feat_dict['Audio'].to(self.device)
                        paudio_wav = feat_dict['PAudio']
                        paudio_feat = self.processor(paudio_wav, sampling_rate=16000, return_tensors="pt", padding='longest').input_values.to(self.device)
                        feature = (video_feat, audio_feat, paudio_feat)
                    else:
                        feature,filenames = batch
                        feature = feature.to(self.device)
                    gpu_tracker.track()
                    model_out = self.model(feature)
                    values = model_out.cpu()
                    gpu_tracker.track()
                    for ind,value in enumerate(values.numpy()):
                        filename = filenames[ind]
                        f.write('{0},{1}\n'.format(filename, value))
                        result_dict[filename] = value
                    track(f'ModelOut:{values}')
                    if iter_id % self.log_step == 0:
                        rate=iter_id/len(dataloader)*100
                        logger('|step {:4d} |total {:4d}| Rate% {:.3f}'.format(iter_id, len(dataloader), rate))
                    track.step()
                    gpu_tracker.step()
        logger("Test Score Prediction Done")
        # print(result_dict)
        return result_dict