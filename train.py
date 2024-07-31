import atexit
import csv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Visible Cuda Devices
from data.data_process import DeepFakeSet
from src.trainer import Trainer, weights_init
from config import get_opt
# from src.VST.video_swin_transformer import VideoClassifier
from src.video_swin_transformer import VideoClassifier
from src.swin_transformer2d import SwinTransformerV2
from src.audioTransformer import Audio2D
from src.ModalFusion import FusionModel
from src.IResNet import InceptionVideoClassifier
import json
from torch import cuda
from src.utils import Logger, load_pretrained, load_pre_fused
import signal
import threading
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import torch


def handle_exit():
    print('Program Killed by signal')

def shut_sub_prog(event: threading.Event):
    event.set()

def train(args, logger):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    processor=None
    if args.modality == 'video':
        # model = VideoClassifier(args, logger, num_classes=1)
        model = InceptionVideoClassifier(args, 1, drop_rate=args.swin_drop)
    elif args.modality == 'audio':
        model = SwinTransformerV2(num_classes=1,embed_dim=128,num_heads=[4,8,16,32 ],depths=[2,2,18,2 ],pretrained_window_sizes=(16,16,16,16))
        weights_init(model)
        load_pretrained(args, model, logger)
    elif args.modality == 'paudio':
        processor = Wav2Vec2Processor.from_pretrained("./checkpoints/wav2vec2-base-960h",local_files_only=True)
        wav_model = Wav2Vec2Model.from_pretrained("./checkpoints/wav2vec2-base-960h")
        model = Audio2D(args, wav_model,num_classes=1)
    elif args.modality == 'fused':
        AudioE = SwinTransformerV2(num_classes=1, use_feat=True, embed_dim=128,num_heads=[4,8,16,32 ],depths=[2,2,18,2 ],pretrained_window_sizes=(16,16,16,16))
        load_pretrained(args, AudioE, logger)
        VideoE = InceptionVideoClassifier(args, 1, drop_rate=args.swin_drop, use_feat=True)
        wav_model = Wav2Vec2Model.from_pretrained("./checkpoints/wav2vec2-base-960h")
        PAudioE = Audio2D(args, wav_model,num_classes=1, use_feat=True)
        processor = Wav2Vec2Processor.from_pretrained("./checkpoints/wav2vec2-base-960h",local_files_only=True)
        model = FusionModel(args, VideoE, AudioE, PAudioE)
    event = threading.Event()
    atexit.register(shut_sub_prog, event)
    data = DeepFakeSet(args, logger=logger)
    data.setup(event)
    device = 'cuda' if cuda.is_available() else 'cpu'
    trainer = Trainer(model, args, device, logger, processor)
    if args.Resume:
        if args.modality == 'fused':
            del model,trainer
            torch.cuda.empty_cache()
            load_pre_fused(args, VideoE, AudioE, PAudioE, logger)
            model = FusionModel(args, VideoE, AudioE, PAudioE)
            trainer = Trainer(model, args, device, logger, processor)
        else:
            trainer.load_ckpt(args)
    if not args.skip_learning:
        trainer.train(data)
    
    # Produce submit.csv
    result = trainer.submit(data)
    fileName='prediction.csv'
    with open(fileName, 'w') as f:
        f.write('video_name,target\n')
        [f.write('{0},{1}\n'.format(key, value)) for key, value in result.items()]

if __name__ == '__main__':
    opt = get_opt()
    logger = Logger(opt.log_dir)
    logger(f'processId: {os.getpid()}')
    logger(f'prarent processId: {os.getppid()}')
    logger(json.dumps(opt.__dict__, indent=4))
    atexit.register(handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    signal.signal(signal.SIGINT, handle_exit)
    train(opt, logger)