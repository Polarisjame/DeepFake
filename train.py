import atexit
import csv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Visible Cuda Devices
import torch
from data.data_process import DeepFakeSet
from src.trainer import Trainer, weights_init
from config import get_opt
# from src.VST.video_swin_transformer import VideoClassifier
from src.video_swin_transformer import VideoClassifier
from src.swin_transformer2d import SwinTransformerV2
from src.audioTransformer import Audio2D
from src.FusionModel import VAModel
from src.IResNet import InceptionVideoClassifier
import json
from torch import cuda
from src.utils import Logger, load_pretrained
import signal
import threading

def handle_exit():
    print('Program Killed by signal')

def shut_sub_prog(event: threading.Event):
    event.set()

def train(args, logger):
    if args.modality == 'video':
        # model = VideoClassifier(args, logger, num_classes=1)
        model = InceptionVideoClassifier(args, 1, drop_rate=args.swin_drop)
    elif args.modality == 'audio':
        model = SwinTransformerV2(num_classes=1,embed_dim=128,num_heads=[4,8,16,32 ],depths=[2,2,18,2 ],pretrained_window_sizes=(16,16,16,16))
        weights_init(model)
        load_pretrained(args, model, logger)
    elif args.modality == 'paudio':
        model = Audio2D(args, num_classes=1)
    elif args.modality == 'fuse':
        AudioE = SwinTransformerV2(num_classes=1,embed_dim=128,num_heads=[4,8,16,32 ],depths=[2,2,18,2 ],pretrained_window_sizes=(16,16,16,16))
        VideoE = VideoClassifier(args, logger, num_classes=1)
        model = VAModel(args, VideoE, AudioE)
    event = threading.Event()
    atexit.register(shut_sub_prog, event)
    data = DeepFakeSet(args, logger=logger)
    data.setup(event)
    device = 'cuda' if cuda.is_available() else 'cpu'
    trainer = Trainer(model, args, device, logger)
    if args.Resume:
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