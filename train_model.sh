#! /bin/bash
export CUDA_VISIBLE_DEVICES=1,2,6
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------DeepFake Training Shell Script--------------------

if true; then
  sample=false
  if ${sample}; then
    data_root='/data/lingfeng/ffdv_phase1_sample'
  else
    data_root='/data/lingfeng/full_data/phase1'
  fi
  modality=fused # video audio paudio fused
  num_frames=32
  num_workers=4

  # pretrain
  video_pretrained_dir='checkpoints/swin_small_patch244_window877_kinetics400_1k.pth'
  audio_pretrained_dir='checkpoints/swinv2_base_patch4_window16_256.pth'
  
  batch_size=8
  accum_step=4
  soft=2.3
  align_loss_rate=1
  log_step=5
  bn_momentum=0.8
  video_pool=mean
  classify_drop=0.2
  swin_drop=0.4
  num_hiddens=256
  l2_decacy=1e-3
  epochs=4
  learning_rate=1e-4
  model_save=500
  random_seed=42
  audio_ckpt_path='checkpoints/VST_deepfake_modalityaudio_batch48_epoch12.pth'
  fused_ckpt_path='checkpoints/VST_deepfake_modalityfused_batch8_epoch3_step2499.pth'

  nohup python3 -u train.py \
    --Resume\
    --soft ${soft}\
    --align_loss_rate ${align_loss_rate}\
    --data_root ${data_root}\
    --bn_momentum ${bn_momentum}\
    --accum_step ${accum_step}\
    --random_seed ${random_seed}\
    --modality ${modality}\
    --num_workers ${num_workers}\
    --num_frames ${num_frames}\
    --video_pretrained_dir ${video_pretrained_dir}\
    --audio_pretrained_dir ${audio_pretrained_dir}\
    --audio_ckpt_path ${audio_ckpt_path}\
    --fused_ckpt_path ${fused_ckpt_path}\
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
    --log_dir logs/DF_Sample:${sample}_Modality:${modality}_Batch:${batch_size}.log\
    >logs/error_out_DF_Sample:${sample}_Modality:${modality}_Batch:${batch_size}.log 2>&1 &
fi

    # --skip_learning\
    # --ckpt_path ${ckpt_path}\
    # --force_generate\
    # --Resume\
    # --val_model\