import torch
from pytorch_lightning import LightningModule
from pytorch_metric_learning import losses

from CLAAD.Source import utils


class Temp_Module(LightningModule):
    def __init__(self,
        f,
        g,
        classifier, 
        opt_config={
            "lr":00.02,

        }
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.classifier = classifier
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.ntx_loss = losses.NTXentLoss()
        self.opt_config = opt_config

    def configure_optimizers(self):
        return torch.optim.Adam(
            [
                {"params": self.f.parameters()},
                {"params": self.g.parameters()},
                {
                    "params": self.classifier.parameters(),
                },
            ],
            **self.opt_config,
        )

    def training_step(self, batch, batch_idx):
        data_dir, labels = batch
        X, Y = utils.apply_transform(data_dir, labels)
        Y_NTXent = torch.arange(X.shape[0])
        Y_NTXent[int(X.shape[0] / 2) :] = Y_NTXent[0 : int(X.shape[0] / 2)]
        h = self.f(X)
        z = self.g(h)
        cls_pred = self.classifier(h)
        loss = self.ntx_loss(z, Y_NTXent) + 0.1 * self.ce_loss(
            cls_pred, Y.long()
        )
        return loss
