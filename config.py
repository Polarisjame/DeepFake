import argparse

def get_opt():
    parser = argparse.ArgumentParser(description="Deepfake")
    
    # DATA
    parser.add_argument('--data_root', type=str, default=r'/data/lingfeng/ffdv_phase1_sample')
    parser.add_argument('--num_frames', type=int, default=32, help='extract fixed number of frames')
    parser.add_argument('-nu', '--num_workers', type=int, default=3, help='thread number')
    
    # Model
    parser.add_argument('--pretrained_dir', type=str, default='checkpoints/swin_base_patch244_window1677_kinetics400_22k_host.pth')
    parser.add_argument('--classify_drop', type=float, default=0.1, help='MLP_dropout_rate')
    parser.add_argument('--swin_drop', type=float, default=0.4, help='VST_dropout_rate')
    parser.add_argument('--num_hiddens', type=int, default=256, help='Hidden Num of Classifier')
    
    # Learning
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='input batch size for training (default: 32)')
    parser.add_argument('-cuda', '--use_cuda', type=bool, default=True, help='Use cuda or not')
    parser.add_argument('--l2_decacy', type=float, default=1e-6)
    parser.add_argument('-e', '--epochs', type=int, default=50, help='input training epoch for training (default: 50)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='input learning rate for training (default: 1e-4)')
    
    # Log
    parser.add_argument('--log_step', type=int, default=10)
    args = parser.parse_args()
    return args