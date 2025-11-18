python train.py \
  --create_stacked \
  --stacked_path "dummy.pt" \
  --dataset_path "archive" \
  --train_hr_glob "Dist Rad Train img/*.*" \
  --train_lr_glob "Dist Rad Train img sim/*.*" \
  --img_size 256 \
  --checkpoints "saved_models/vggloss_GAN_generator_dist_tib.pth"
  