from collections import OrderedDict
import os
import random
import threading
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
# import moviepy.editor as mp
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from torch import nn
from pydub import AudioSegment

REF_SEC=5

def extract_frames(video_path, num_frames, target_size, trans):
    frames = torch.zeros(0,3,target_size,target_size, dtype=torch.float32)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = total_frames // num_frames
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)) 
            image = trans(image)
            frames=torch.concat((frames, image.unsqueeze(0)), 0)
        else:
            break
    
    cap.release()
    return frames

def extract_wav(video_path):
    path_id = random.randint(0,10000)
    audio_path = f'./extract_wav/extracted_audio_{path_id}.wav'
    
    track = AudioSegment.from_file(video_path, "mp4")
    file_handle = track.set_frame_rate(16000).export(audio_path, format="wav")
    y, sr = librosa.load(audio_path, sr=16000)
    feature = y
    return feature

# def preprocess_frames(frames,target_size):
#     preprocessed_frames = []
#     for frame in frames:
#         # 调整帧的大小
#         frame = cv2.resize(frame, (target_size,target_size))
#         # 归一化
#         frame = frame / 255.0
#         # print(frame,frame.shape)
#         preprocessed_frames.append(frame)
    
#     return np.array(preprocessed_frames)

def generate_mel_spectrogram(video_path, thread_id=0, n_mels=128, fmax=8000, target_size=(224, 224)):
    # 提取音频
    audio_path = f'extracted_audio_{thread_id}.wav'
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

def split_audio_wav(video_path, extract_audio_wav_path):
    video = mp.VideoFileClip(video_path)
    target_dir = os.path.join(extract_audio_wav_path,video_path.split('/')[-1][:-4] + '.wav')
    video.audio.write_audiofile(target_dir, verbose=False, logger=None)
    
def split_audio_wav_thread(sub_lists, extract_audio_wav_path, logger):
    thread_id = threading.current_thread().ident + random.randint(1, 1000)
    for index,video_path in enumerate(sub_lists):
        target_dir = os.path.join(extract_audio_wav_path,video_path.split('/')[-1][:-4] + '.wav')
        if os.path.exists(target_dir):
            continue
        if index % 100 == 0:
            logger(f"Thread:{thread_id} Processed {index} Files")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(target_dir, verbose=False, logger=None)

def process_list(items, event: threading.Event, extract_video_tensor_path, logger, num_frames):
    # 创建5个线程，每个线程处理一个列表项
    thread_id = threading.current_thread().ident + random.randint(1, 1000)
    for index,video_path in enumerate(items):
        if event.is_set():
            print(f"Tread{thread_id} Stoped Unexpected")
            break
        target_dir = os.path.join(extract_video_tensor_path,video_path.split('/')[-1][:-4] + '.pt')
        if os.path.exists(target_dir):
            continue
        if index % 100 == 0:
            logger(f"Thread:{thread_id} Processed {index} Files")
        video_feature_row = extract_frames(video_path, extract_video_tensor_path,num_frames, 224)
        # video_feature = torch.tensor(video_feature,dtype=torch.float32).view(3,-1,224,224)
        # np.save(target_dir,video_feature)

def generate_sample_wave(video_path, 
                         sample_rate=22050,
                         stft_length=0.032,
                         stft_step=0.016,
                         mel_bins=80,
                         mel_lower_edge_hertz=80.,
                         mel_upper_edge_hertz=7600.,
                         mel_num_bins=40):
    audio_path = f'extracted_audio_A.wav'
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # 加载音频文件
    y, sr = librosa.load(audio_path, sr=sample_rate) # 22050*seconds
    if len(y) < sample_rate*REF_SEC:
        pad_length = sample_rate*REF_SEC - len(y)
        audio = np.pad(y, (0, pad_length), mode='constant')
    else:
        audio = y[:sample_rate*REF_SEC]
    mask = (audio != 0)
    true_index = mask.nonzero()
    stft_length = int(sample_rate * stft_length)
    stft_step = int(sample_rate * stft_step)
    tensor_y = torch.tensor(audio, dtype=torch.float32)
    stfts = tf.signal.stft(tensor_y[true_index],
                        frame_length=stft_length,
                        frame_step=stft_step,
                        fft_length=stft_length,
                        pad_end=True)
    spectrogram = tf.abs(stfts)
    num_spectrogram_bins = spectrogram.shape[-1]
    mel_sample_rate = sample_rate
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        mel_num_bins, num_spectrogram_bins, mel_sample_rate,
        mel_lower_edge_hertz, mel_upper_edge_hertz)
    spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    spectrogram.set_shape(spectrogram.shape[:-1] +
                        linear_to_mel_weight_matrix.shape[-1:])
    spectrogram = torch.from_numpy(spectrogram.numpy())
    len_pad = 314-spectrogram.shape[0]
    mask_pad = torch.zeros(314)
    mask_pad[[i for i in range(spectrogram.shape[0])]]=1
    pad_dim = spectrogram.shape[1]
    pad_zro = torch.zeros((len_pad,pad_dim))
    spectrogram = torch.concat((spectrogram, pad_zro),dim=0)
    return spectrogram,mask_pad
    # mask = torch.tensor(mask).unsqueeze(0)
    
def collate_opt(batch):
    features, labels, filenames = zip(*batch)
    pre_list = []
    label = torch.stack(labels)
    for feature in features:
        pre_list.append(feature)
    return pre_list, label, filenames
        
def fusion_collate(batch):
    features, labels, filenames = zip(*batch)
    feat_res_dict = {}
    labels = torch.stack(labels)
    # labels = torch.as_tensor(labels, dtype=torch.int)
    for feat_dict in features:
        for k,v in feat_dict.items():
            if k != 'PAudio':
                v = v.unsqueeze(0)
                if k in feat_res_dict.keys():
                    feat_res_dict[k] = torch.cat((feat_res_dict[k],v),dim=0)
                else:
                    feat_res_dict[k] = v
            else:
                if k in feat_res_dict.keys():
                    feat_res_dict[k].append(v)
                else:
                    feat_res_dict[k] = [v]
    return feat_res_dict, labels, filenames

class Drawer(object):
    def __init__(self, modality, phase):
        self.reset()
        self.modality = modality
        self.phase = phase
        
    def reset(self):
        self.log_list = []

    def update(self, val, n=1):
        self.log_list.append(val)
        
    def draw(self, epoch):
        length = len(self.log_list)
        plt.plot([i for i in range(length)],self.log_list)
        plt.savefig(f"./checkpoints/Modality:{self.modality}_Phase:{self.phase}_Epoch{epoch}.png")
        plt.show()
        
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
        
def load_pre_fused(args, VE, AE, PAE, logger=None):
    device = torch.device('cpu')
    if args.audio_ckpt_path is not None:
        logger(f"==============> Loading weight {args.audio_ckpt_path} for Audio fine-tuning......")
        checkpoint = torch.load(args.audio_ckpt_path, map_location=torch.device('cpu'))['checkpoint']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if "head" in k:
                continue
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        AE.load_state_dict(new_state_dict,strict=False) 
        logger(f"=> loaded successfully '{args.audio_ckpt_path}'")
    if args.video_ckpt_path is not None:
        logger(f"==============> Loading weight {args.video_ckpt_path} for Audio fine-tuning......")
        checkpoint = torch.load(args.video_ckpt_path, map_location=torch.device('cpu'))['checkpoint']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        VE.load_state_dict(new_state_dict) 
        logger(f"=> loaded successfully '{args.video_ckpt_path}'")
    if args.paudio_ckpt_path is not None:
        logger(f"==============> Loading weight {args.paudio_ckpt_path} for Audio fine-tuning......")
        checkpoint = torch.load(args.paudio_ckpt_path, map_location=torch.device('cpu'))['checkpoint']
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        PAE.load_state_dict(new_state_dict) 
        logger(f"=> loaded successfully '{args.paudio_ckpt_path}'")
        
def load_pretrained(config, model, logger):
    logger(f"==============> Loading weight {config.audio_pretrained_dir} for fine-tuning......")
    checkpoint = torch.load(config.audio_pretrained_dir, map_location=torch.device('cpu'))
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
    # head_bias_pretrained = state_dict['head.bias']
    # Nc1 = head_bias_pretrained.shape[0]
    # Nc2 = model.head.bias.shape[0]
    # if (Nc1 != Nc2):
    #     if Nc1 == 21841 and Nc2 == 1000:
    #         logger("loading ImageNet-22K weight to ImageNet-1K ......")
    #         map22kto1k_path = f'data/map22kto1k.txt'
    #         with open(map22kto1k_path) as f:
    #             map22kto1k = f.readlines()
    #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
    #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
    #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
    #     else:
    #         torch.nn.init.constant_(model.head.bias, 0.)
    #         torch.nn.init.constant_(model.head.weight, 0.)
    #         del state_dict['head.weight']
    #         del state_dict['head.bias']
    #         logger(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger(msg)

    logger(f"=> loaded successfully '{config.audio_pretrained_dir}'")

    del checkpoint
    torch.cuda.empty_cache()
    
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True