# This module provides the main entry point for the BSSC program.
# it includes the following blocks: inference step, logging callbacks,
# train step, validation step, test step, and main function for the training with pytorch.
# Author: Tsien-LL
# Date: 2023-03-19
# Version: 1.0
# ==============================================================================
# ==============================================================================
# import packages
import argparse
import logging
import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

log = logging.getLogger(os.path.basename(__file__))
try:
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(package_dir)
except NameError:
    pass
# write a training loop for pytorch with ddp


def init_training(args):
    import timm
    from models.wavlm import WavLM, WavLMConfig
    from utils.ddp import DdpSetter
    from utils.logger import AverageMeter, TrainLogging

    model = timm.create_model("wavlm_large", pretrained=True)

    temp_path = "/mnt/nas/public2/lilong/data/checkpoints/saved/nas/saved/pretrained/WavLM-Large.pt"
    checkpoint = torch.load(temp_path, map_location="cpu")
    cfg = WavLMConfig(checkpoint["cfg"])
    print("config", cfg)

    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)


def inference_step(model, data, input_keys, output_keys=None):
    # inference step
    pass


def logging_callbacks():
    # logging callbacks
    pass


def train_step(model, data, optimizer):
    # train step
    pass


def validation_step(model, data):
    # validation step
    pass


def train_loop():
    # train loop
    pass


def main():
    # main function for the training with pytorch
    pass
