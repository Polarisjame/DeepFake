import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Visible Cuda Devices

from data.data_process import DeepFakeSet
from src.trainer import Trainer
from config import get_opt
# from src.VST.video_swin_transformer import VideoClassifier
from src.video_swin_transformer import VideoClassifier
from src.swin_transformer2d import SwinTransformerV2
import json
from torch import cuda
from src.utils import Logger, load_pretrained


def train(args, logger):
    if args.modality == 'video':
        model = VideoClassifier(args)
    elif args.modality == 'audio':
        model = SwinTransformerV2(num_classes=2, pretrained_window_sizes=(16,16,16,16))
        load_pretrained(args, model, logger)
    data = DeepFakeSet(args, logger=logger)
    data.setup()
    device = 'cuda' if cuda.is_available() else 'cpu'
    trainer = Trainer(model, args, device, logger)
    trainer.train(data)

if __name__ == '__main__':
    opt = get_opt()
    logger = Logger(opt.log_dir)
    logger(f'processId: {os.getpid()}')
    logger(f'prarent processId: {os.getppid()}')
    logger(json.dumps(opt.__dict__, indent=4))
    train(opt, logger)