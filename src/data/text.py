"""
Text data utilities for CV-KAN.

Includes:
- Simple tokenizer and vocabulary
- Text dataset wrapper
- Collate function with padding
- SST-2 loader
"""

import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    """Basic whitespace and punctuation tokenizer."""

    def __init__(self, lower: bool = True):
        self.lower = lower
        # Split on whitespace and punctuation
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        return self.pattern.findall(text)


class Vocabulary:
    """Map words to indices."""

    def __init__(self, specials: list[str] | None = None):
        if specials is None:
            specials = ["<pad>", "<unk>"]
        self.specials = specials
        self.stoi = {s: i for i, s in enumerate(specials)}
        self.itos = {i: s for i, s in enumerate(specials)}

    def build_from_texts(self, texts: list[str], min_freq: int = 1):
        """Build vocabulary from list of text strings."""
        counter = Counter()
        tokenizer = SimpleTokenizer()

        for text in texts:
            tokens = tokenizer.tokenize(text)
            counter.update(tokens)

        # Add frequent words
        idx = len(self.itos)
        for word, count in counter.most_common():
            if count >= min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

        print(f"Built vocabulary with {len(self.itos)} tokens")

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.stoi["<unk>"])


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: Vocabulary,
        max_len: int = 64,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = self.tokenizer.tokenize(text)
        # Truncate
        if len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]

        # Convert to indices
        indices = [self.vocab[t] for t in tokens]

        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "length": len(indices),
        }


def pad_collate(batch):
    """Collate function that pads sequences."""
    indices = [item["indices"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    lengths = torch.tensor([item["length"] for item in batch])

    # Pad
    indices_padded = torch.nn.utils.rnn.pad_sequence(indices, batch_first=True, padding_value=0)

    # Create mask (1 for valid, 0 for pad)
    mask = (indices_padded != 0).long()

    return {
        "indices": indices_padded,
        "mask": mask,
        "label": labels,
        "lengths": lengths,
    }


def download_sst2(root_dir: str = "data/SST-2"):
    """Download SST-2 dataset."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    tsv_path = root / "train.tsv"
    dev_path = root / "dev.tsv"

    if tsv_path.exists() and dev_path.exists():
        return

    print("Downloading SST-2...")
    url = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
    zip_path = root / "SST-2.zip"

    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(root.parent)

    print("Download complete.")


def load_sst2(
    root_dir: str = "data/SST-2",
    max_len: int = 64,
    min_freq: int = 2,
    vocab: Vocabulary | None = None,
):
    """Load SST-2 data."""
    download_sst2(root_dir)
    root = Path(root_dir)

    def read_tsv(path):
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        # Skip header
        rows = [line.split("\t") for line in lines[1:]]
        texts = [r[0] for r in rows]
        labels = [int(r[1]) for r in rows]
        return texts, labels

    train_texts, train_labels = read_tsv(root / "train.tsv")
    dev_texts, dev_labels = read_tsv(root / "dev.tsv")

    # Build vocab if not provided
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_from_texts(train_texts, min_freq=min_freq)

    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocab, max_len)

    return train_dataset, dev_dataset, vocab


# ============================================================================
# IMDB Dataset (Longer sequences, binary sentiment)
# ============================================================================

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download_imdb(root_dir: str = "data/imdb"):
    """Download IMDB dataset."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    data_path = root / "aclImdb"
    if data_path.exists():
        return

    import tarfile

    tar_path = root / "aclImdb_v1.tar.gz"
    if not tar_path.exists():
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(IMDB_URL, tar_path)

    print("Extracting IMDB...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(root)
    print("IMDB ready.")


def load_imdb(
    root_dir: str = "data/imdb",
    max_len: int = 256,
    min_freq: int = 5,
    vocab: Vocabulary | None = None,
):
    """
    Load IMDB movie review dataset.

    Binary sentiment classification with longer sequences than SST-2.
    """
    download_imdb(root_dir)
    root = Path(root_dir) / "aclImdb"

    def read_split(split_dir):
        texts, labels = [], []
        for label_name, label_id in [("pos", 1), ("neg", 0)]:
            label_dir = split_dir / label_name
            for file_path in label_dir.glob("*.txt"):
                text = file_path.read_text(encoding="utf-8")
                texts.append(text)
                labels.append(label_id)
        return texts, labels

    train_texts, train_labels = read_split(root / "train")
    test_texts, test_labels = read_split(root / "test")

    # Build vocab if not provided
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_from_texts(train_texts, min_freq=min_freq)

    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)

    return train_dataset, test_dataset, vocab


# ============================================================================
# AG News Dataset (Topic classification, 4 classes)
# ============================================================================

AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"


def download_agnews(root_dir: str = "data/agnews"):
    """Download AG News dataset."""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    train_path = root / "train.csv"
    test_path = root / "test.csv"

    if train_path.exists() and test_path.exists():
        return

    print("Downloading AG News dataset...")
    for filename in ["train.csv", "test.csv"]:
        url = AG_NEWS_URL + filename
        try:
            urllib.request.urlretrieve(url, root / filename)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            # Try alternative source
            alt_url = f"https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/{filename}"
            urllib.request.urlretrieve(alt_url, root / filename)
    print("AG News ready.")


def load_agnews(
    root_dir: str = "data/agnews",
    max_len: int = 128,
    min_freq: int = 2,
    vocab: Vocabulary | None = None,
):
    """
    Load AG News topic classification dataset.

    4 classes: World, Sports, Business, Sci/Tech.
    """
    download_agnews(root_dir)
    root = Path(root_dir)

    import csv

    def read_csv(path):
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                # Format: label, title, description
                label = int(row[0]) - 1  # Convert 1-4 to 0-3
                text = row[1] + " " + row[2]  # Combine title and description
                texts.append(text)
                labels.append(label)
        return texts, labels

    train_texts, train_labels = read_csv(root / "train.csv")
    test_texts, test_labels = read_csv(root / "test.csv")

    # Build vocab if not provided
    if vocab is None:
        vocab = Vocabulary()
        vocab.build_from_texts(train_texts, min_freq=min_freq)

    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)

    return train_dataset, test_dataset, vocab
