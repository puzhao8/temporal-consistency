import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from temporal_consistency import train_conv_lstm

def run():
    torch.multiprocessing.freeze_support()
    print('torch.multiprocessing.freeze_support()')

@hydra.main(config_path="./config", config_name="config")
def run_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


    # # For reproducibility, set random seed
    if cfg.experiment.seed == 'None':
        cfg.experiment.seed = random.randint(1, 10000)
    random.seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)
    torch.manual_seed(cfg.experiment.seed)
    torch.cuda.manual_seed_all(cfg.experiment.seed)
    
    wandb.init(config=cfg, project=cfg.project.name, name=cfg.experiment.name, entity=cfg.experiment.wandb_team)


    train_conv_lstm(cfg)

if __name__ == '__main__':
    run()
    run_app()

    