import os
import sys
import torch
import logging
import pandas as pd
from PIL import Image
from datetime import datetime
from argparse import Namespace


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')

        labels = self.data_info.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels).float()


def log_arguments(args: Namespace, logger: logging.Logger):
    logger.info("\nArguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('\n')


def setup_logger(log_filename, model_name):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        sys.exit(1)

    # Add date and time to the log file name
    base_name, ext = os.path.splitext(log_filename)
    if ext == '':
        ext = '.log'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_filename = f"{base_name}_{model_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, full_log_filename)

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    try:
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {e}")
        sys.exit(1)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def min_max_normalize(features):
    min_val = torch.min(features)
    max_val = torch.max(features)
    return (features - min_val) / (max_val - min_val)


def save_checkpoint(model, checkpoint_name, model_name):
    # Create the checkpoints directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{checkpoint_name}.pth")

    # Save only the state dictionary
    torch.save(model.state_dict(), checkpoint_path)

    return checkpoint_path


def compute_time(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days:.0f} days, {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds"
    elif hours > 0:
        return f"{hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds"
    elif minutes > 0:
        return f"{minutes:.0f} minutes, {seconds:.2f} seconds"
    else:
        return f"{seconds:.2f} seconds"