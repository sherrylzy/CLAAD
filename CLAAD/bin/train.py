import argparse

from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything

from CLAAD.datamodule.base import Audio_DataModule
from CLAAD.dataset.mimii import mimii_dataset_builder
from CLAAD.module.temp import Temp_Module
from CLAAD.Network.LinearClassifier import LinCLS
from CLAAD.Network.ProjectionHead import Projection
from CLAAD.Network.ResNet18 import resnet18

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/example.yaml")
args = parser.parse_args()
cfg = OmegaConf.load(args.config)
seed_everything(42)

module = Temp_Module(
    f=resnet18(),
    g=Projection(),
    classifier=LinCLS(),
    opt_config=cfg["opt"],
)
train_dataset, val_dataset = mimii_dataset_builder(**cfg["dataset"])
datamodule = Audio_DataModule(
    train_dataset, val_dataset, dataloader_conf=cfg["dataloader"]
)

trainer = Trainer(accelerator="auto", logger=False)
trainer.fit(module, datamodule=datamodule)
