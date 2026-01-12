import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
from torchvision import transforms, utils

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Latent-Retinex Diffusion Models')
    parser.add_argument("--config", default='unsupervised copy.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--mode', type=str, default='evaluation', help='training or evaluation')
    parser.add_argument('--resume', default='/home/ubuntu/797979/project1/LightenDiffusion-main-rcb/results/rsam/LOL/model_best_val_ema_pgfm_ceg.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')#/home/ubuntu/project1/LightenDiffusion-main/ckpt1/avg_top4.pth.tar
    parser.add_argument("--image_folder", default='/home/ubuntu/797979/project1/LightenDiffusion-main-rcb/results/ceshi/LOL', type=str,
                        help="Location to save restored images")
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    print("=> using dataset '{}'".format(config.data.val_dataset))
    DATASET = datasets.__dict__[config.data.type](config)
    _, val_loader = DATASET.get_loaders()

    # create model
    print("=> creating denoising-diffusion model")
    diffusion = DenoisingDiffusion(args, config)
    diffusion.load_ddm_ckpt(args.resume, ema=True)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader)


if __name__ == '__main__':
    main()


    
# # 示例：用你原来的 best ckpt 做评估
# CUDA_VISIBLE_DEVICES=0 python evaluate.py \
#   --config unsupervised copy.yml \
#   --resume /home/ubuntu/797979/project1/LightenDiffusion-main-rcb/results/rsam/LOL/model_best_val_ema.pth.tar \
#   --image_folder /home/ubuntu/797979/project1/LightenDiffusion-main-rcb/results/rsam/1
