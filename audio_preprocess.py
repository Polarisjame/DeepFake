from config import get_opt
from src.utils import Logger,  GpuInfoTracker, generate_mel_spectrogram
import json
import os
import cv2

if __name__ == 'main':
    args = get_opt()
    logger = Logger(args.log_dir)
    logger(f'processId: {os.getpid()}')
    logger(f'prarent processId: {os.getppid()}')
    logger(json.dumps(args.__dict__, indent=4))
    
    root = args.data_root
    extract_audio_img_path = os.path.join(root, 'TestAudioImgs')
    dataset_path = os.path.join(root , 'phase2', 'testset1seen')
    filepaths = [os.path.join(dataset_path, video) for video in os.listdir(dataset_path)]

    if not os.path.exists(extract_audio_img_path):
        os.mkdir(extract_audio_img_path)
    logger("Processing Audio File!")

    for index, video_path in enumerate(filepaths):
        target_dir = os.path.join(extract_audio_img_path,video_path.split('/')[-1][:-4] + '.jpg')
        if os.path.exists(target_dir):
            continue
        if index % 100 == 0:
            rate = int(index/len(filepaths)*100)
            logger("Test: ["+"*"*rate+"-"*(100-rate)+"]"+f" ({index}/{len(filepaths)})")
        mel_spectrogram_image = generate_mel_spectrogram(video_path)
        cv2.imwrite(target_dir, mel_spectrogram_image)