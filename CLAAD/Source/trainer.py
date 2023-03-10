import argparse

import torch
from omegaconf import OmegaConf
from pytorch_metric_learning import losses
from tqdm import tqdm

from CLAAD.Dataset.train_test import train_test
from CLAAD.Network.LinearClassifier import LinCLS
from CLAAD.Network.ProjectionHead import Projection
from CLAAD.Network.ResNet18 import resnet18
from CLAAD.Source import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_prepare(lr):
    f = resnet18().to(device)
    g = Projection().to(device)
    CLS = LinCLS().to(device)
    opt = torch.optim.Adam(
        [
            {"params": f.parameters()},
            {"params": g.parameters()},
            {
                "params": CLS.parameters(),
            },
        ],
        lr=lr,
    )
    return f, g, CLS, opt


def trainer(
    Train,
    NTXent=losses.NTXentLoss(),
    CELoss=torch.nn.CrossEntropyLoss(),
    num_epochs=2,
    verbosity=0,  # epoch 200
):
    f, g, CLS, opt = model_prepare()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batch = 0
        # for train_features, train_labels in Train:
        print(f"start epoch {epoch}")
        for data_dir, labels in tqdm(Train.dataset):
            # X, Y = utils.apply_transform(train_features, train_labels)
            X, Y = utils.apply_transform(data_dir, labels)
            opt.zero_grad()
            X = X.to(device)
            Y = Y.to(device)
            Y_NTXent = torch.arange(X.shape[0])
            Y_NTXent[int(X.shape[0] / 2) :] = Y_NTXent[0 : int(X.shape[0] / 2)]
            h = f(X)
            z = g(h)
            cls_pred = CLS(h)
            loss = NTXent(z, Y_NTXent) + 0.1 * CELoss(cls_pred, Y.long())

            loss.backward()
            opt.step()
            epoch_loss += loss
            # epoch_loss += NTXent(z,Y_NTXent)
            # cls_loss += CELoss(cls_pred, Y.long())
            num_batch += 1
        if verbosity > 0:
            print(
                "epoch : {}/{}, loss = {:.6f}".format(
                    epoch + 1, num_epochs, epoch_loss / num_batch
                )
            )


"""testing trainer program"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/example.yaml")
args = parser.parse_args()
conf = OmegaConf.load(args.config)

Train, Test = train_test(
    train_size=conf["train_size"], device=conf["machine"], id=0
)
trainer(Train)
