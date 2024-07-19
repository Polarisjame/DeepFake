import atexit
import csv
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Visible Cuda Devices

from data.data_process import DeepFakeSet
from src.trainer import Trainer
from config import get_opt
# from src.VST.video_swin_transformer import VideoClassifier
from src.video_swin_transformer import VideoClassifier
from src.swin_transformer2d import SwinTransformerV2
from src.audioTransformer import Audio2D
import json
from torch import cuda
from src.utils import Logger, load_pretrained
import signal

def handle_exit():
    print('Program Killed by signal')


def train(args, logger):
    if args.modality == 'video':
        model = VideoClassifier(args)
    elif args.modality == 'audio':
        model = SwinTransformerV2(num_classes=2,embed_dim=128,num_heads=[4,8,16,32 ],depths=[2,2,18,2 ],pretrained_window_sizes=(16,16,16,16))
        load_pretrained(args, model, logger)
    elif args.modality == 'paudio':
        model = Audio2D(args, num_classes=2)
    data = DeepFakeSet(args, logger=logger)
    data.setup()
    device = 'cuda' if cuda.is_available() else 'cpu'
    trainer = Trainer(model, args, device, logger)
    trainer.train(data)
    
    # Produce submit.csv
    # result = trainer.submit(data)
    # fileName='prediction.csv'
    # with open(fileName,"wb") as csv_file:
    #     writer=csv.writer(csv_file)
    #     for key,value in result.items:
    #         writer.writerow([key,value])

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