from __future__ import annotations

import os

import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

EXCHANGE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/exchange_rate.csv"
)


class ExchangeDataset(Dataset):
    def __init__(
        self,
        root: str = "./data/exchange",
        flag: str = "train",
        target: str | None = None,
        scale: bool = True,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        download: bool = True,
        subset_size: int | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.target = target
        self.root_path = root
        self.data_path = "exchange_rate.csv"
        self.subset_size = subset_size

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        if download:
            self.download()
        self.__read_data__()

    def download(self) -> None:
        os.makedirs(self.root_path, exist_ok=True)
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            print(f"Downloading Exchange dataset from {EXCHANGE_URL}...")
            try:
                response = requests.get(EXCHANGE_URL, allow_redirects=True, timeout=30)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print("Download complete.")
            except Exception as exc:
                raise RuntimeError(f"Failed to download exchange dataset: {exc}") from exc

    def __read_data__(self) -> None:
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        data = df_raw.iloc[:, 1:]  # drop date column

        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        borders = [(0, train_end), (train_end, val_end), (val_end, n)]

        border1, border2 = borders[self.set_type]

        if self.target:
            data = data[[self.target]]

        if self.scale:
            train_data = data.iloc[borders[0][0] : borders[0][1]]
            self.scaler.fit(train_data.values)
            data_values = self.scaler.transform(data.values)
        else:
            data_values = data.values

        self.data_x = data_values[border1:border2]
        self.data_y = data_values[border1:border2]

        if self.subset_size:
            self.data_x = self.data_x[: self.subset_size]
            self.data_y = self.data_y[: self.subset_size]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def create_exchange_dataloader(
    root: str = "./data/exchange",
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 96,
    num_workers: int = 0,
    subset_size: int | None = None,
):
    train_set = ExchangeDataset(
        root=root,
        flag="train",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size,
    )
    val_set = ExchangeDataset(
        root=root,
        flag="val",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size // 10 if subset_size else None,
    )
    test_set = ExchangeDataset(
        root=root,
        flag="test",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size // 10 if subset_size else None,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_set.data_x.shape[1]
