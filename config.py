import argparse

def get_opt():
    parser = argparse.ArgumentParser(description="Deepfake")
    
    # DATA
    parser.add_argument('--skip_learning', action='store_true', help='skip train stage and get submission')
    parser.add_argument('--data_root', type=str, default=r'/data/lingfeng/full_data/phase1')
    parser.add_argument('--modality', type=str, default='audio')
    parser.add_argument('--num_frames', type=int, default=32, help='extract fixed number of frames')
    parser.add_argument('--force_generate', action='store_true', help='force process audio file')
    parser.add_argument('-nu', '--num_workers', type=int, default=1, help='thread number')
    
    # Model
    parser.add_argument('--video_pretrained_dir', type=str, default='checkpoints/swin_small_patch244_window877_kinetics400_1k.pth')
    parser.add_argument('--audio_pretrained_dir', type=str, default='checkpoints/swinv2_tiny_patch4_window16_256.pth')
    parser.add_argument('--classify_drop', type=float, default=0.1, help='MLP_dropout_rate')
    parser.add_argument('--swin_drop', type=float, default=0.1, help='VST_dropout_rate')
    parser.add_argument('--num_hiddens', type=int, default=128, help='Hidden Num of Classifier')
    parser.add_argument('--video_pool', type=str, help='VST Pool Method')
    parser.add_argument('--audio_ckpt_path',type=str,default=None)
    parser.add_argument('--video_ckpt_path',type=str,default=None)
    parser.add_argument('--paudio_ckpt_path',type=str,default=None)
    parser.add_argument('--Resume', action='store_true', help='resume model from ckpt')

    # Learning
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='input batch size for training (default: 32)')
    parser.add_argument('-cuda', '--use_cuda', type=bool, default=True, help='Use cuda or not')
    parser.add_argument('--l2_decacy', type=float, default=0.05)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='input training epoch for training (default: 50)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='input learning rate for training (default: 1e-4)')
    parser.add_argument('--model_save', type=int, default=5, help='save model per %d round')
    
    # Log
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()
    return args