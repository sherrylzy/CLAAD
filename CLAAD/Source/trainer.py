import torch
from Dataset.train_test import train_test
from Network.LinearClassifier import LinCLS
from Network.ProjectionHead import Projection
from Network.ResNet18 import resnet18
from pytorch_metric_learning import losses
from Source import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(
    Train,
    f=resnet18().to(device),
    g=Projection().to(device),
    CLS=LinCLS().to(device),
    NTXent=losses.NTXentLoss(),
    CELoss=torch.nn.CrossEntropyLoss(),
    num_epochs=2,
    verbosity=0,  # epoch 200
    pre_train=False,
):
    if not pre_train:
        optimizer = torch.optim.Adam(
            [
                {"params": f.parameters()},
                {"params": g.parameters()},
                {
                    "params": CLS.parameters(),
                },
            ],
            lr=1e-2,
        )

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batch = 0
            # for train_features, train_labels in Train:
            for data_dir, labels in Train.dataset:
                # X, Y = utils.apply_transform(train_features, train_labels)
                X, Y = utils.apply_transform(data_dir, labels)
                optimizer.zero_grad()
                X = X.to(device)
                Y = Y.to(device)
                Y_NTXent = torch.arange(X.shape[0])
                Y_NTXent[int(X.shape[0] / 2) :] = Y_NTXent[
                    0 : int(X.shape[0] / 2)
                ]
                h = f(X)
                z = g(h)
                cls_pred = CLS(h)
                loss = NTXent(z, Y_NTXent) + 0.1 * CELoss(cls_pred, Y.long())

                loss.backward()
                optimizer.step()
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
    else:
        PATH = "state_dict_model.pt"
        f.load_state_dict(torch.load(PATH))


"""testing trainer program"""
Train, Test = train_test(train_size=0.75, device="valve", id=0)
trainer(Train)
