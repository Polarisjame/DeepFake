from data.data_process import DeepFakeSet
from src.trainer import Trainer
from config import get_opt
from src.video_swin_transformer import VideoClassifier
import json
from torch import cuda
import os
from src.utils import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3' # Visible Cuda Devices

def train(args, logger):
    model = VideoClassifier(args)
    data = DeepFakeSet(args)
    data.setup()
    device = 'cuda' if cuda.is_available() else 'cpu'
    trainer = Trainer(model, args, device, logger)
    trainer.train(data)

if __name__ == '__main__':
    opt = get_opt()
    logger = Logger(f'./logs/deepFake_lr{opt.learning_rate:.1e}_batch{opt.batch_size}.log')
    logger(f'processId: {os.getpid()}')
    logger(f'prarent processId: {os.getppid()}')
    logger(json.dumps(opt.__dict__, indent=4))
    train(opt, logger)