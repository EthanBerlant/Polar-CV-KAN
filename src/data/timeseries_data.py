import os

import pandas as pd
import requests
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"


class ETTh1Dataset(Dataset):
    def __init__(
        self,
        root: str = "./data/ETT",
        flag: str = "train",
        size: int = None,
        subset_size: int = None,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        download: bool = True,
    ):
        # features: M=multivariate, S=univariate
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # Init
        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale

        self.root_path = root
        self.data_path = "ETTh1.csv"

        if download:
            self.download()

        self.subset_size = subset_size
        self.__read_data__()

    def download(self):
        os.makedirs(self.root_path, exist_ok=True)
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            print(f"Downloading ETTh1 from {ETTH1_URL}...")
            try:
                r = requests.get(ETTH1_URL, allow_redirects=True)
                with open(file_path, "wb") as f:
                    f.write(r.content)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download ETTh1: {e}")

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Split: Train: 12 months, Val: 4 months, Test: 4 months
        # ETTh1 is hourly.
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


def create_timeseries_dataloader(
    root: str = "./data/ETT",
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 96,
    num_workers: int = 0,
    subset_size: int = None,
):
    train_set = ETTh1Dataset(
        root=root,
        flag="train",
        size=None,
        features="M",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size,
    )
    val_set = ETTh1Dataset(
        root=root,
        flag="val",
        size=None,
        features="M",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size // 10 if subset_size else None,
    )
    test_set = ETTh1Dataset(
        root=root,
        flag="test",
        size=None,
        features="M",
        seq_len=seq_len,
        pred_len=pred_len,
        download=True,
        subset_size=subset_size // 10 if subset_size else None,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_set.data_x.shape[1]


# ============================================================================
# ETTm1 Dataset (15-minute granularity)
# ============================================================================

ETTM1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv"


class ETTm1Dataset(Dataset):
    """
    ETTm1 dataset: Electricity Transformer Temperature at 15-minute intervals.
    Higher frequency than ETTh1, good for testing temporal resolution sensitivity.
    """

    def __init__(
        self,
        root: str = "./data/ETT",
        flag: str = "train",
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        download: bool = True,
        subset_size: int = None,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root
        self.data_path = "ETTm1.csv"

        if download:
            self.download()
        self.subset_size = subset_size
        self.__read_data__()

    def download(self):
        os.makedirs(self.root_path, exist_ok=True)
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            print(f"Downloading ETTm1 from {ETTM1_URL}...")
            try:
                r = requests.get(ETTM1_URL, allow_redirects=True)
                with open(file_path, "wb") as f:
                    f.write(r.content)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download ETTm1: {e}")

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ETTm1 splits: Train 12 months, Val 4 months, Test 4 months
        # 15-min intervals = 4 * 24 = 96 per day
        border1s = [0, 12 * 30 * 96 - self.seq_len, 12 * 30 * 96 + 4 * 30 * 96 - self.seq_len]
        border2s = [12 * 30 * 96, 12 * 30 * 96 + 4 * 30 * 96, 12 * 30 * 96 + 8 * 30 * 96]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def create_ettm1_dataloader(
    root: str = "./data/ETT",
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 96,
    num_workers: int = 0,
    subset_size: int = None,
):
    """Create ETTm1 dataloaders (15-minute granularity)."""
    train_set = ETTm1Dataset(
        root=root, flag="train", seq_len=seq_len, pred_len=pred_len, subset_size=subset_size
    )
    val_set = ETTm1Dataset(
        root=root,
        flag="val",
        seq_len=seq_len,
        pred_len=pred_len,
        subset_size=subset_size // 10 if subset_size else None,
    )
    test_set = ETTm1Dataset(
        root=root,
        flag="test",
        seq_len=seq_len,
        pred_len=pred_len,
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


# ============================================================================
# Weather Dataset
# ============================================================================

# WEATHER_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/weather/weather.csv" # Broken
WEATHER_URL = "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather.csv"


class WeatherDataset(Dataset):
    """
    Weather dataset: 21 meteorological features recorded every 10 minutes.
    Features include temperature, humidity, wind speed, etc.
    Good for multivariate forecasting with diverse feature types.
    """

    def __init__(
        self,
        root: str = "./data/weather",
        flag: str = "train",
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 96,
        download: bool = True,
        subset_size: int = None,
    ):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        assert flag in ["train", "val", "test"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root
        self.data_path = "weather.csv"

        if download:
            self.download()
        self.subset_size = subset_size
        self.__read_data__()

    def download(self):
        os.makedirs(self.root_path, exist_ok=True)
        file_path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(file_path):
            print(f"Downloading Weather dataset from {WEATHER_URL}...")
            try:
                r = requests.get(WEATHER_URL, allow_redirects=True)
                with open(file_path, "wb") as f:
                    f.write(r.content)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download Weather: {e}")

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Weather has ~52K samples. Use 70/10/20 split
        n = len(df_raw)
        border1s = [0, int(n * 0.7) - self.seq_len, int(n * 0.8) - self.seq_len]
        border2s = [int(n * 0.7), int(n * 0.8), n]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]  # Skip date column
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

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

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


def create_weather_dataloader(
    root: str = "./data/weather",
    batch_size: int = 32,
    seq_len: int = 96,
    pred_len: int = 96,
    num_workers: int = 0,
    subset_size: int = None,
):
    """Create Weather dataloaders (21 meteorological features)."""
    train_set = WeatherDataset(
        root=root, flag="train", seq_len=seq_len, pred_len=pred_len, subset_size=subset_size
    )
    val_set = WeatherDataset(
        root=root,
        flag="val",
        seq_len=seq_len,
        pred_len=pred_len,
        subset_size=subset_size // 10 if subset_size else None,
    )
    test_set = WeatherDataset(
        root=root,
        flag="test",
        seq_len=seq_len,
        pred_len=pred_len,
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
