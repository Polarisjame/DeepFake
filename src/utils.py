import os
import threading
import cv2
import numpy as np
from datetime import datetime
import moviepy.editor as mp
import torch
import librosa

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

def generate_mel_spectrogram(video_path, n_mels=128, fmax=8000, target_size=(224, 224)):
    # 提取音频
    audio_path = 'extracted_audio.wav'
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # 加载音频文件
    y, sr = librosa.load(audio_path)

    # 生成MEL频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # 将频谱图转换为dB单位
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 归一化到0-255之间
    S_dB_normalized = cv2.normalize(S_dB, None, 0, 255, cv2.NORM_MINMAX)
    
    # 将浮点数转换为无符号8位整型
    S_dB_normalized = S_dB_normalized.astype(np.uint8)

    # 缩放到目标大小
    img_resized = cv2.resize(S_dB_normalized, target_size, interpolation=cv2.INTER_LINEAR)

    return img_resized

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
        print(datetime.now(), string, file=self.f, flush=True)
        
    def __delete__(self):
        self.f.close()
        
def load_pretrained(config, model, logger):
    logger(f"==============> Loading weight {config.audio_pretrained_dir} for fine-tuning......")
    checkpoint = torch.load(config.audio_pretrained_dir, map_location='cuda:0')
    state_dict = checkpoint['model']
    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger(msg)

    logger(f"=> loaded successfully '{config.audio_pretrained_dir}'")

    del checkpoint
    torch.cuda.empty_cache()

def process_list(items,extract_audio_img_path):
    # 创建5个线程，每个线程处理一个列表项
    for video_path in items:
        mel_spectrogram_image = generate_mel_spectrogram(video_path)
        cv2.imwrite(os.path.join(extract_audio_img_path,video_path.split('/')[-1][:-4] + '.jpg'), mel_spectrogram_image)