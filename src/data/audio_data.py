import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# On Windows Python 3.8+, we need to explicitly add FFmpeg DLL directory
# before importing torchaudio/torchcodec
if sys.platform == "win32":
    # Common FFmpeg installation paths
    ffmpeg_paths = [
        os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "ffmpeg-8.0.1-full_build-shared",
            "bin",
        ),
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
    ]
    for path in ffmpeg_paths:
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except (AttributeError, OSError):
                pass  # add_dll_directory not available or path invalid

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, root: str = "./data", download: bool = False):
        super().__init__(root, download=download)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [
                    os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj
                ]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


class LogMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512):
        super().__init__()
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, x):
        return torch.log(self.mel_spectrogram(x) + 1e-9)


def pad_sequence(batch):
    # Batch is list of tuples (waveform, sample_rate, label, speaker_id, utterance_number)
    # We only need waveform and label
    waveforms = [item[0].squeeze() for item in batch]
    labels = [item[2] for item in batch]

    # Pad waveforms to max length in batch
    max_len = max(w.shape[0] for w in waveforms)
    padded_waveforms = []
    for w in waveforms:
        if w.shape[0] < max_len:
            padding = max_len - w.shape[0]
            padded_w = torch.nn.functional.pad(w, (0, padding))
        else:
            padded_w = w
        padded_waveforms.append(padded_w.unsqueeze(0))

    return torch.stack(padded_waveforms), labels


class SpeechCommandsCollate:
    def __init__(self, label_to_index):
        self.label_to_index = label_to_index

    def __call__(self, batch):
        tensors, targets = [], []

        # Determine max length in this batch for padding
        max_len = 0
        for waveform, _, _label, *_ in batch:
            max_len = max(max_len, waveform.size(1))

        for waveform, _, label, *_ in batch:
            # Pad
            if waveform.size(1) < max_len:
                padding = max_len - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))

            tensors += [waveform]
            targets += [torch.tensor(self.label_to_index[label])]

        tensors = torch.stack(tensors)

        # Optional: Transform to Spectrogram here or in model.
        # The plan says "use_stft_frontend: True" in model, implying model takes waveform or STFT.
        # CVKANAudio doc says it takes waveform or spectrogram.
        # We'll return raw waveform and let model handle it or specific training script handle it.

        targets = torch.stack(targets)
        return tensors, targets


def create_audio_dataloader(
    root: str = "./data/speech_commands",
    batch_size: int = 64,
    subset_size: int = None,
    download: bool = True,
    num_workers: int = 2,
):
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for audio benchmarking.")

    os.makedirs(root, exist_ok=True)

    # Check if we can create dataset
    try:
        train_set = SubsetSC("training", root=root, download=download)
        val_set = SubsetSC("validation", root=root, download=download)
        test_set = SubsetSC("testing", root=root, download=download)
    except Exception as e:
        print(f"Error loading SpeechCommands: {e}")
        # Build empty dummy if failed (to avoid script crash during structure check)
        # But for benchmarking it will fail.
        raise e

    # Create label map
    # Convert subset to list to safely iterate multiple times if needed, though Set should be fine.
    # The error came from "set(datapoint[2]...)" failing inside sorted().
    # Let's iterate explicitly to debug or be safe.
    all_labels = set()
    for i in range(len(train_set)):
        _, _, label, *_ = train_set[i]
        all_labels.add(label)

    labels = sorted(list(all_labels))
    label_to_index = {label: index for index, label in enumerate(labels)}

    if subset_size:
        indices = torch.randperm(len(train_set))[:subset_size]
        train_set = Subset(train_set, indices)

        indices = torch.randperm(len(val_set))[: subset_size // 10]
        val_set = Subset(val_set, indices)

        indices = torch.randperm(len(test_set))[: subset_size // 10]
        test_set = Subset(test_set, indices)

    # Use num_workers for parallel loading (2 is safe on Windows)
    # n_workers = 2 if sys.platform != "win32" else 0  # Windows often has multiprocessing issues
    n_workers = num_workers

    collate_fn = SpeechCommandsCollate(label_to_index)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=n_workers,
        pin_memory=True,
        persistent_workers=(n_workers > 0),
    )

    return train_loader, val_loader, test_loader, len(labels)


# ============================================================================
# UrbanSound8K Dataset
# ============================================================================

URBANSOUND8K_URL = "https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz"


class UrbanSound8KDataset(torch.utils.data.Dataset):
    """
    UrbanSound8K dataset for environmental sound classification.

    10 classes: air_conditioner, car_horn, children_playing, dog_bark,
    drilling, engine_idling, gun_shot, jackhammer, siren, street_music

    Note: This dataset requires manual download due to size (~6GB).
    Download from: https://urbansounddataset.weebly.com/urbansound8k.html
    """

    CLASSES = [
        "air_conditioner",
        "car_horn",
        "children_playing",
        "dog_bark",
        "drilling",
        "engine_idling",
        "gun_shot",
        "jackhammer",
        "siren",
        "street_music",
    ]

    def __init__(self, root: str, fold: int = None, transform=None, download: bool = False):
        """
        Args:
            root: Root directory containing UrbanSound8K folder
            fold: Fold number (1-10) for cross-validation, or None for all folds
            transform: Optional transform to apply to waveforms
            download: If True, attempt to download (prints instructions)
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        audio_dir = self.root / "UrbanSound8K" / "audio"
        metadata_path = self.root / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"

        if not audio_dir.exists():
            if download:
                print(f"\nUrbanSound8K dataset not found at {self.root}")
                print("Please download manually from:")
                print("https://urbansounddataset.weebly.com/urbansound8k.html")
                print(f"Extract to: {self.root}")
            raise FileNotFoundError(f"UrbanSound8K not found at {audio_dir}")

        # Load metadata
        import csv

        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if fold is None or int(row["fold"]) == fold:
                    audio_path = audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
                    if audio_path.exists():
                        self.samples.append(
                            {
                                "path": audio_path,
                                "label": int(row["classID"]),
                                "fold": int(row["fold"]),
                            }
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sample_rate = torchaudio.load(sample["path"])

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample["label"]


def urbansound8k_collate_fn(batch):
    waveforms, labels = zip(*batch, strict=False)
    max_len = max(w.size(1) for w in waveforms)

    padded = []
    for w in waveforms:
        if w.size(1) < max_len:
            w = F.pad(w, (0, max_len - w.size(1)))
        padded.append(w)

    return torch.stack(padded), torch.tensor(labels)


def create_urbansound8k_dataloader(
    root: str = "./data/urbansound8k",
    batch_size: int = 64,
    test_fold: int = 10,
    subset_size: int = None,
    download: bool = False,
    num_workers: int = 2,
):
    """
    Create UrbanSound8K dataloaders using 10-fold cross-validation.

    Args:
        root: Data directory
        batch_size: Batch size
        test_fold: Fold to use for testing (1-10)
        subset_size: If set, limit samples per fold
        download: Whether to attempt download (prints instructions)

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for UrbanSound8K.")

    # Use test_fold for test, (test_fold-1) for val, rest for train
    val_fold = test_fold - 1 if test_fold > 1 else 10
    train_folds = [f for f in range(1, 11) if f not in [test_fold, val_fold]]

    # Combine train folds
    train_samples = []
    for fold in train_folds:
        try:
            ds = UrbanSound8KDataset(root, fold=fold, download=download)
            train_samples.extend(ds.samples)
        except FileNotFoundError:
            raise

    # Create datasets
    train_set = UrbanSound8KDataset(root, fold=None)
    train_set.samples = train_samples

    val_set = UrbanSound8KDataset(root, fold=val_fold)
    test_set = UrbanSound8KDataset(root, fold=test_fold)

    if subset_size:
        train_set.samples = train_set.samples[:subset_size]
        val_set.samples = val_set.samples[: subset_size // 10]
        test_set.samples = test_set.samples[: subset_size // 10]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=urbansound8k_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=urbansound8k_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=urbansound8k_collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, 10


# ============================================================================
# ESC-50 Dataset
# ============================================================================

ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"


class ESC50Dataset(torch.utils.data.Dataset):
    """
    ESC-50: Environmental Sound Classification with 50 classes.

    Categories include: Animals, Natural sounds, Human non-speech,
    Interior/domestic sounds, Exterior/urban sounds

    Note: Smaller than UrbanSound8K, auto-downloadable.
    """

    def __init__(self, root: str, fold: int = None, transform=None, download: bool = True):
        """
        Args:
            root: Root directory
            fold: Fold number (1-5) for cross-validation, or None for all
            transform: Optional transform
            download: Whether to download if not present
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        audio_dir = self.root / "ESC-50-master" / "audio"
        meta_path = self.root / "ESC-50-master" / "meta" / "esc50.csv"

        if not audio_dir.exists():
            if download:
                self._download()
            else:
                raise FileNotFoundError(f"ESC-50 not found at {audio_dir}")

        # Load metadata
        import csv

        with open(meta_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if fold is None or int(row["fold"]) == fold:
                    audio_path = audio_dir / row["filename"]
                    if audio_path.exists():
                        self.samples.append(
                            {
                                "path": audio_path,
                                "label": int(row["target"]),
                                "fold": int(row["fold"]),
                                "category": row["category"],
                            }
                        )

    def _download(self):
        """Download and extract ESC-50 dataset."""
        import urllib.request
        import zipfile

        self.root.mkdir(parents=True, exist_ok=True)
        zip_path = self.root / "esc50.zip"

        if not zip_path.exists():
            print("Downloading ESC-50 dataset...")
            urllib.request.urlretrieve(ESC50_URL, zip_path)

        print("Extracting ESC-50...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(self.root)
        print("ESC-50 ready.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sample_rate = torchaudio.load(sample["path"])

        # ESC-50 is already 44.1kHz, resample to 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample["label"]


def esc50_collate_fn(batch):
    waveforms, labels = zip(*batch, strict=False)
    max_len = max(w.size(1) for w in waveforms)

    padded = []
    for w in waveforms:
        if w.size(1) < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.size(1)))
        padded.append(w)

    return torch.stack(padded), torch.tensor(labels)


def create_esc50_dataloader(
    root: str = "./data/esc50",
    batch_size: int = 64,
    test_fold: int = 5,
    subset_size: int = None,
    download: bool = True,
    num_workers: int = 2,
):
    """
    Create ESC-50 dataloaders using 5-fold cross-validation.

    Args:
        root: Data directory
        batch_size: Batch size
        test_fold: Fold to use for testing (1-5)
        subset_size: If set, limit samples
        download: Whether to download dataset

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio is required for ESC-50.")

    # Use test_fold for test, (test_fold-1) for val, rest for train
    val_fold = test_fold - 1 if test_fold > 1 else 5
    train_folds = [f for f in range(1, 6) if f not in [test_fold, val_fold]]

    # Combine train folds
    train_samples = []
    for fold in train_folds:
        ds = ESC50Dataset(root, fold=fold, download=download)
        train_samples.extend(ds.samples)

    # Create datasets
    train_set = ESC50Dataset(root, fold=None, download=download)
    train_set.samples = train_samples

    val_set = ESC50Dataset(root, fold=val_fold, download=download)
    test_set = ESC50Dataset(root, fold=test_fold, download=download)

    if subset_size:
        train_set.samples = train_set.samples[:subset_size]
        val_set.samples = val_set.samples[: subset_size // 10]
        test_set.samples = test_set.samples[: subset_size // 10]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=esc50_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=esc50_collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=esc50_collate_fn,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, 50
