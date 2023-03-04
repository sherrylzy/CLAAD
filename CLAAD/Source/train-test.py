import torch
from Dataset.data_loader import MIMII, audiodir
from torch.utils.data import ConcatDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_test(train_size, device="valve", id=0):
    train_size = 0.75
    dir, label = audiodir(device, id)
    dir_abnormal, label_abnormal = audiodir(device, id, Data="abnormal")

    dataset_normal = MIMII(dir, label)
    dataset_abnormal = MIMII(dir_abnormal, label_abnormal)
    train_dataset, test_normal_dataset = torch.utils.data.random_split(
        dataset_normal,
        [
            int(len(dataset_normal) * train_size),
            len(dataset_normal) - int(len(dataset_normal) * train_size),
        ],
    )

    test_dataset = ConcatDataset([test_normal_dataset, dataset_abnormal])

    Train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    Test = torch.utils.data.DataLoader(
        test_dataset, batch_size=300, shuffle=True, num_workers=4
    )
    return Train, Test
