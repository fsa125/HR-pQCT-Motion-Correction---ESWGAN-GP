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

def build_train_loader(args):
    """Only used when we want to (re)create the stacked .pt file."""
    hr_shape = (args.img_size, args.img_size)
    train_paths_hr = glob.glob(os.path.join(args.dataset_path, args.train_hr_glob))
    train_paths_lr = glob.glob(os.path.join(args.dataset_path, args.train_lr_glob))

    if len(train_paths_hr) == 0 or len(train_paths_lr) == 0:
        raise FileNotFoundError(
            f"No training images found.\n"
            f"HR pattern: {os.path.join(args.dataset_path, args.train_hr_glob)}\n"
            f"LR pattern: {os.path.join(args.dataset_path, args.train_lr_glob)}"
        )

    ds = ImageDataset(train_paths_hr, train_paths_lr, hr_shape=hr_shape)
    # You originally used batch_size=1 for building the stacked dataset
    return DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
       
    )

# -----------------------
# Training (stacked_dataset style)
# -----------------------
def train(args):
    warnings.filterwarnings("ignore")
    set_seed(args.seed)
    device = args.device
    cuda = (device.type == "cuda")

    # (Optional) build & store stacked dataset
    if args.create_stacked:
        print(f"[Info] Creating stacked dataset from loader and saving to {args.stacked_path} ...")
        train_loader = build_train_loader(args)
        Store_train_data(train_loader, args.stacked_path)
        print("[Info] Stacked dataset saved.")

    # Load stacked dataset (.pt) — expected to be a list where each item is a TensorDataset-like
    # object with .tensors (lr, hr)
    if not os.path.isfile(args.stacked_path):
        raise FileNotFoundError(
            f"stacked dataset file not found: {args.stacked_path}\n"
            f"Run with --create_stacked to generate it."
        )
    stacked_dataset = torch.load(args.stacked_path)
    n_items = len(stacked_dataset)
    if n_items == 0:
        raise RuntimeError("Loaded stacked_dataset is empty.")

    # Models
    channels = 1
    hr_shape = (args.img_size, args.img_size)

    generator = UNetAttGenerator(in_channels=channels, out_channels=channels).to(device)
    discriminator = DiscriminatorATT(input_shape=(channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()  # inference mode for content features

    if args.checkpoints:
        print(f"[Info] Loading pre-trained generator weights from {args.checkpoints} ...")

        generator.load_state_dict(torch.load(args.checkpoints))

    # Losses
    criterion_content = torch.nn.L1Loss().to(device)
    criterion_pix = torch.nn.MSELoss().to(device)  # optional pixel term if enabled

    # Optims
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Tensor alias (matches your style)
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Logs
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    train_gen_losses, train_disc_losses = [], []

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        generator.train()
        discriminator.train()

        # === YOUR SAMPLING STYLE: pick a random example each iteration ===
        random_integer = random.randint(0, n_items - 1)
        # Expect each entry to have .tensors = (imgs_lr, imgs_hr)
        imgs_lr = stacked_dataset[random_integer].tensors[0].to(device)
        imgs_hr = stacked_dataset[random_integer].tensors[1].to(device)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad(set_to_none=True)

        gen_hr = generator(imgs_lr)

        # WGAN-GP adversarial term: maximize D(gen) => minimize -D(gen)
        loss_g_adv = -discriminator(gen_hr).mean()

        # Feature/content term
        with torch.no_grad():
            real_features = feature_extractor(imgs_hr)
        gen_features = feature_extractor(gen_hr)
        loss_content = criterion_content(gen_features, real_features)

        # Optional pixel term
        if args.lambda_pix > 0.0:
            loss_pix = criterion_pix(imgs_hr, gen_hr)
        else:
            loss_pix = torch.tensor(0.0, device=device)

        # Total G loss (mirrors your original: content + lambda_adv * adv + optional pixel)
        loss_G = loss_content + args.lambda_adv * loss_g_adv + args.lambda_pix * loss_pix
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad(set_to_none=True)

        d_fake = discriminator(gen_hr.detach()).mean()
        d_real = discriminator(imgs_hr).mean()
        gp = gradient_penalty(discriminator, imgs_hr, gen_hr.detach(), device)

        loss_D = d_fake - d_real + args.gp_weight * gp
        loss_D.backward()
        optimizer_D.step()

        train_gen_losses.append(loss_G.item())
        train_disc_losses.append(loss_D.item())

        if epoch % args.log_every == 0:
            pbar.write(f"[Epoch {epoch}/{args.epochs}] "
                       f"G: {loss_G.item():.4f} | D: {loss_D.item():.4f}")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            g_path = os.path.join(args.save_dir, f"generator_epoch_{epoch}.pth")
            d_path = os.path.join(args.save_dir, f"discriminator_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)
            pbar.write(f"Saved: {g_path} | {d_path}")

    # Final save
    torch.save(generator.state_dict(), os.path.join(args.save_dir, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.save_dir, "discriminator_final.pth"))


# -----------------------
# Argparse
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train UNetAttGenerator + DiscriminatorATT (WGAN-GP + content) using stacked_dataset sampling."
    )
    # Data building (only for creating the stacked .pt)
    parser.add_argument("--dataset_path", type=str, default="archive/",
                        help="Root path containing training folders (used only with --create_stacked).")
    parser.add_argument("--train_hr_glob", type=str, default="Dist Rad Train img/*.*",
                        help="Glob (relative to dataset_path) for HR training images (only when creating stacked file).")
    parser.add_argument("--train_lr_glob", type=str, default="Dist Rad Train img sim/*.*",
                        help="Glob (relative to dataset_path) for LR images (only when creating stacked file).")
    parser.add_argument("--img_size", type=int, default=256,
                        help="Image size expected by your model and dataset.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (only when creating stacked file).")

    # Stacked dataset I/O
    parser.add_argument("--stacked_path", type=str, default="dummy.pt",
                        help="Path to the stacked dataset .pt file.")
    parser.add_argument("--create_stacked", action="store_true",
                        help="Create the stacked .pt from the dataloader before training.")

    # Optimization
    parser.add_argument("--epochs", type=int, default=300000,
                        help="Number of training iterations (one sample per epoch).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for both G and D.")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="Adam beta1.")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="Adam beta2.")

    # Loss weights
    parser.add_argument("--lambda_adv", type=float, default=1e-3,
                        help="Weight for adversarial term in G loss.")
    parser.add_argument("--lambda_pix", type=float, default=0.0,
                        help="Weight for pixel-wise MSE term in G loss (0 to disable).")
    parser.add_argument("--gp_weight", type=float, default=0.2,
                        help="Gradient penalty weight for WGAN-GP.")

    # Misc
    parser.add_argument("--save_dir", type=str, default="saved_models",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--save_every", type=int, default=150000,
                        help="Save checkpoints every N epochs.")
    parser.add_argument("--log_every", type=int, default=1000,
                        help="Log every N epochs.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")
    
    parser.add_argument("--checkpoints", type=str, default = "checkpoints",
                        help="Load the checkpoints from a directory")

    args = parser.parse_args()
    args.device = get_device(force_cpu=args.cpu)
    return args

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()