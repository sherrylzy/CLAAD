from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class Audio_DataModule(LightningDataModule):
    def __init__(self, 
        train_dataset,
        val_dataset,
        dataloader_conf={
            "batch_size":2,
            "shuffle": True,
            "num_workers":4,
            "drop_last": True,
        },
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.dataloader_conf = dataloader_conf

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_conf)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_conf)