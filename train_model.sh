#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,4,5

# -------------------DeepFake Training Shell Script--------------------

if true; then
  sample=false
  if ${sample}; then
    data_root='/data/lingfeng/ffdv_phase1_sample'
  else
    data_root='/data/lingfeng/full_data/phase1'
  fi
  modality=video # video audio paudio
  num_frames=32

  # pretrain
  video_pretrained_dir='checkpoints/swin_small_patch244_window877_kinetics400_1k.pth'
  video_pretrained_cfg='./configs/recognition/swin/swin_small_patch244_window877_kinetics400_1k.py'
  audio_pretrained_dir='checkpoints/swinv2_base_patch4_window16_256.pth'
  
  video_pool=Attention
  classify_drop=0.2
  swin_drop=0.4
  num_hiddens=256
  batch_size=24
  l2_decacy=0.001
  epochs=50
  learning_rate=3e-4
  model_save=10
  log_step=50

  nohup python3 -u train.py \
    --data_root ${data_root}\
    --modality ${modality}\
    --num_frames ${num_frames}\
    --video_pretrained_dir ${video_pretrained_dir}\
    --video_pretrained_cfg ${video_pretrained_cfg}\
    --audio_pretrained_dir ${audio_pretrained_dir}\
    --classify_drop ${classify_drop}\
    --swin_drop ${swin_drop}\
    --num_hiddens ${num_hiddens}\
    --batch_size ${batch_size}\
    --l2_decacy ${l2_decacy}\
    --epochs ${epochs}\
    --learning_rate ${learning_rate}\
    --model_save ${model_save}\
    --log_step ${log_step}\
    --video_pool ${video_pool}\
    --log_dir logs/DF_Sample:${sample}_Modality:${modality}_Batch:${batch_size}TEST.log\
    >logs/error_out_DF_Sample:${sample}_Modality:${modality}_Batch:${batch_size}.log 2>&1 &
fi

    # --force_generate\