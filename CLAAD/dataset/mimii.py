from glob import glob

import torch
import torchaudio
from torch.utils.data import ConcatDataset, Dataset


def mimii_file_list_generator(
    path,
    prefix_normal="normal",
    prefix_anormal="abnormal",
    ext="wav",
):
    files_list = []
    for prefix in [prefix_normal, prefix_anormal]:
        glob_str = f"{path}/{prefix}/*.{ext}"
        files_list.append(glob(glob_str))
    return files_list


class Mimii_Dataset(Dataset):
    def __init__(self, data_dir, labels):
        self.data_dir = data_dir
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.data_dir[idx]
        x, sr = torchaudio.load(path)
        x = x.mean(axis=0)
        y = self.labels[idx]
        return x, y


def mimii_dataset_builder(path, normal_ratio_in_test=0.75):
    normal_files, abnormal_files = mimii_file_list_generator(path)
    normal_labels = [True] * len(normal_files)
    abnormal_labels = [False] * len(abnormal_files)
    dataset_normal = Mimii_Dataset(normal_files, normal_labels)
    dataset_abnormal = Mimii_Dataset(abnormal_files, abnormal_labels)

    normal_size_in_test = int(len(dataset_normal) * normal_ratio_in_test)
    train_dataset, test_normal_dataset = torch.utils.data.random_split(
        dataset_normal,
        [
            normal_size_in_test,
            len(dataset_normal) - normal_size_in_test,
        ],
    )
    test_dataset = ConcatDataset([test_normal_dataset, dataset_abnormal])
    return train_dataset, test_dataset
