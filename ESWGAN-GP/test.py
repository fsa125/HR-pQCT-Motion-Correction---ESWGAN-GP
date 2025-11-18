#!/usr/bin/env python3
import os
import glob
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import randn
from torchmetrics.image import VisualInformationFidelity
from image_functions import *
# --- your modules ---
from utils import *
from ImageDataset import ImageDataset
from preloading import Store_train_data
from GradientPenalty import gradient_penalty
from model import FeatureExtractor, UNetAttGenerator, DiscriminatorATT

# -----------------------
# Helpers
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_test_loader(args):
   
    hr_shape = (args.img_size, args.img_size)
    test_paths_hr = glob.glob(os.path.join(args.dataset_path, args.test_hr_glob))
    test_paths_lr = glob.glob(os.path.join(args.dataset_path, args.test_lr_glob))

    if len(test_paths_hr) == 0 or len(test_paths_lr) == 0:
        raise FileNotFoundError(
            f"No testing images found.\n"
            f"HR pattern: {os.path.join(args.dataset_path, args.test_hr_glob)}\n"
            f"LR pattern: {os.path.join(args.dataset_path, args.test_lr_glob)}"
        )

    ds = ImageDataset(test_paths_hr, test_paths_lr, hr_shape=hr_shape)
    # You originally used batch_size=1 for building the stacked dataset
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
       
    )

# -----------------------
# Testing 
# -----------------------
def test(args):
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    device = args.device
    cuda = (device.type == "cuda")

    # (Optional) build & store stacked dataset
    

    # Models
    channels = 1
    hr_shape = (args.img_size, args.img_size)

   
    generator = UNetAttGenerator(in_channels=channels, out_channels=channels).to(device)

   



    #generator = UNetAttGenerator(in_channels=channels, out_channels=channels).to(device)
    #discriminator = DiscriminatorATT(input_shape=(channels, *hr_shape)).to(device)
    #feature_extractor = FeatureExtractor().to(device)
    #feature_extractor.eval()  # inference mode for content features

    if args.checkpoints:
        print(f"[Info] Loading pre-trained generator weights from {args.checkpoints} ...")

        generator.load_state_dict(torch.load(args.checkpoints))

   

    # Tensor alias (matches your style)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Logs
    Path(args.results).mkdir(parents=True, exist_ok=True)
    

    
    
    





    # Set font style and size globally
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'Times New Roman',
        'font.size': 12
    })

    epoch = 100
    tqdm_bar = tqdm(build_test_loader(args), desc=f'Testing Epoch {epoch} ', total=int(len(build_test_loader(args))))
    #results = []
    #img_list = []
    for batch_idx, imgs in enumerate(tqdm_bar):
            generator.eval()
            # Configure model input
            imgs_lr = Variable(imgs["lr"].type(Tensor)) # motion-corrputed data
            imgs_hr = Variable(imgs["hr"].type(Tensor)) # ground truth
            # Adversarial ground truths
        
            ### Eval Generator
            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)
            gen_hr = normalize_tensor(gen_hr)
            t_imgs_hr = normalize_tensor(imgs_hr)
            imgs_lr = normalize_tensor(imgs_lr)
        
            img_lr_np = imgs_lr.detach().cpu().numpy()
            img_lr_np = np.transpose(img_lr_np, (0, 2, 3, 1))
            img_hr_np = gen_hr.detach().cpu().numpy()
            img_hr_np = np.transpose(img_hr_np, (0, 2, 3, 1))
            #img_list.append(img_hr_np)
            img_thr_np = t_imgs_hr.detach().cpu().numpy()
            img_thr_np = np.transpose(img_thr_np, (0, 2, 3, 1))
        
        
            fig, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=300)  # Create a figure and a set of subplots with 1 row and 2 columns
            # Plot the first image on the first subplot
            axes[0].imshow(histogram_matching(img_thr_np[0],img_lr_np[0]), cmap='gray')
            axes[0].set_title('Simulated Motion \nCorrupted Image')
            axes[0].axis('off')
   
            # Plot the second image on the second subplot
            axes[1].imshow(histogram_matching(img_thr_np[0],img_hr_np[0]), cmap='gray')
            axes[1].set_title('Predicted Motion \nCompensated Image')
            axes[1].axis('off')
        
            img = axes[2].imshow(img_thr_np[0], cmap='gray')
            axes[2].set_title('Ground Truth Image')
            axes[2].axis('off')
        
            difference_image = show_difference(img_thr_np[0], histogram_matching(img_thr_np[0],img_hr_np[0]))
        
            img = axes[3].imshow(difference_image, cmap= 'plasma')
            axes[3].set_title('Difference image')
            axes[3].axis('off')
    
        # Add colorbar to the difference image
            cbar = fig.colorbar(img, ax=axes[3], fraction=0.046, pad= 0.04)
            cbar.set_label('Difference intensity')

            fig.savefig('results/Pap_Fig{}.pdf'.format(batch_idx))
            plt.show()
        

# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Test and store the images in a folder."
    )
    # Data building (only for creating the stacked .pt)
    parser.add_argument("--dataset_path", type=str, default="archive/",
                        help="Root path containing testing folders.")
    parser.add_argument("--test_hr_glob", type=str, default="Dist Rad Test img/*.*",
                        help="Glob (relative to dataset_path) for HR testing images.")
    parser.add_argument("--test_lr_glob", type=str, default="Dist Rad Test img sim/*.*",
                        help="Glob (relative to dataset_path) for LR testing images.")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size expected by your model and dataset.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (only when creating stacked file).")


    # Misc
    parser.add_argument("--results", type=str, default="results",
                        help="Directory to save images.")
    
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    
    parser.add_argument("--checkpoints", type=str, default = "checkpoints",
                        help="Load the checkpoints from a directory")
    
  

    args = parser.parse_args()
    args.device = get_device(force_cpu=args.cpu)
    return args

def main():
    args = parse_args()
    test(args)

if __name__ == "__main__":
    main()