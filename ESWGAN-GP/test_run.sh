python test.py \
  --dataset_path "archive" \
  --test_hr_glob "Dist Rad Test img/*.*" \
  --test_lr_glob "Dist Rad Test img sim/*.*" \
  --img_size 256 \
  --results "results" \
  --checkpoints "saved_models/generator_final.pth"