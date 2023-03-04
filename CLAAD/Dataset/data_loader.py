import os

import torch
import torchaudio
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audiodir(
    machine, id, Data="normal", base_dir="/home/qmpzzpmq/CLAAD/MIMII/"
):
    # /content/drive/MyDrive/SADCL/Dataset/'
    """
    Find the audio directory
    Inputs:
    machine: Name of the machine (valve/slider/fan/pump)
    id: ID of the machine (0,2,4,6)
    base_dir = Base directory of the dataset

    Outputs:
    dir = List of data adresses
    label = List of labels (0 -> normal, 1 -> abnormal)
    """
    normaldir = (
        base_dir + machine + "/id_" + str(format(id, "02d")) + "/normal"
    )
    abnormaldir = (
        base_dir + machine + "/id_" + str(format(id, "02d")) + "/abnormal"
    )
    dir = []
    label = []
    if Data == "normal":
        list = os.listdir(normaldir)
        for i in list:
            dir_address = normaldir + "/" + i
            dir.append(dir_address)
            label.append(0)

    else:
        list = os.listdir(abnormaldir)
        for i in list:
            dir_address = abnormaldir + "/" + i
            dir.append(dir_address)
            label.append(1)

    return dir, label


class MIMII(Dataset):
    def __init__(self, data_dir, labels):
        self.labels = labels
        self.data_dir = data_dir

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.data_dir[idx]
        x, sr = torchaudio.load(path)
        x = x.mean(axis=0)
        y = self.labels[idx]
        return x, y
