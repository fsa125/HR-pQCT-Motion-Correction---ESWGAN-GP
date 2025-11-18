@echo off
REM ==============================
REM Run WGAN-GP stacked dataset training
REM ==============================

python train.py ^
  --create_stacked ^
  --stacked_path "dummy.pt" ^
  --epochs 3024000 ^
  --lambda_adv 1e-3 ^
  --gp_weight 0.2 ^
  --save_dir saved_models_dist_tib ^
  --save_every 150000 ^
  --log_every 1000

pause