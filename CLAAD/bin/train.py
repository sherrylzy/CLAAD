import argparse

import torch
from omegaconf import OmegaConf
from pytorch_metric_learning import losses
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything

from CLAAD.Dataset.train_test import train_test
from CLAAD.Network.LinearClassifier import LinCLS
from CLAAD.Network.ProjectionHead import Projection
from CLAAD.Network.ResNet18 import resnet18
from CLAAD.Source import utils
from CLAAD.module.temp import Temp_Module

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/example.yaml")
args = parser.parse_args()
conf = OmegaConf.load(args.config)
seed_everything(42)

train_dl, test_dl = train_test(
    train_size=conf["train_size"], device=conf["machine"], id=0
)
module = Temp_Module(
    f=resnet18(),
    g=Projection(),
    classifier=LinCLS(),
    lr=conf["lr"],
)
trainer = Trainer(accelerator="auto", logger=False)
trainer.fit(module, train_dl, test_dl)
