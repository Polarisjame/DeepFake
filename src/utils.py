import cv2
import numpy as np
from datetime import datetime

def extract_frames(video_path, num_frames):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定提取帧的间隔
    frame_interval = total_frames // num_frames
    
    frames = []
    for i in range(num_frames):
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    return frames

def preprocess_frames(frames,target_size):
    preprocessed_frames = []
    for frame in frames:
        # 调整帧的大小
        frame = cv2.resize(frame, (target_size,target_size))
        # 归一化
        frame = frame / 255.0
        # print(frame,frame.shape)
        preprocessed_frames.append(frame)
    
    return np.array(preprocessed_frames)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count
          
class Logger():
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.f = open(log_dir, 'a')
        self.f.truncate(0)
    
    def __call__(self, string):
        print(datetime.now(), string, file=self.f)
        
    def __delete__(self):
        self.f.close()