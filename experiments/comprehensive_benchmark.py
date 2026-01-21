import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Models
# Baselines
from experiments.baselines.nlp_baseline import BiLSTMBaseline, TinyTransformerBaseline
from src.data.audio_data import (
    create_audio_dataloader,
    create_esc50_dataloader,
    create_urbansound8k_dataloader,
)
from src.data.image_data import (
    create_cifar10_dataloader,
    create_fashionmnist_dataloader,
)
from src.data.image_extension import create_tinyimagenet_dataloader

# Assuming other baselines are importable or standard (can facilitate imports via factory)
# Loaders
from src.data.nlp_loader import NLPDataLoader
from src.data.timeseries_data import (
    create_timeseries_dataloader,
    create_weather_dataloader,
)
from src.data.timeseries_extension import create_exchange_dataloader
from src.models.cv_kan_audio import CVKANAudio

# Assuming generic CVKAN works for other domains or specific classes exist
from src.models.cv_kan_image import CVKANImageClassifier
from src.models.cv_kan_nlp import CVKANNLP
from src.models.cv_kan_timeseries import CVKANTimeSeries

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveBenchmark:
    """
    Executes the comprehensive benchmark across 4 domains, 12 datasets.
    """

    DOMAINS = {
        "image": ["CIFAR10", "FashionMNIST", "TinyImageNet"],
        "audio": ["SpeechCommands", "ESC50", "UrbanSound8K"],
        "timeseries": ["ETTh1", "Weather", "Exchange"],
        "nlp": ["IMDB", "AG_NEWS"],  # SST5 currently aliased/mocked in loader
    }

    BASELINES = {
        "nlp": ["BiLSTM", "TinyTransformer"],
        "image": ["ResNet18", "ViTTiny"],
        "audio": ["CNN", "AST"],
        "timeseries": ["LSTM", "DLinear"],
    }

    def __init__(
        self,
        output_dir="outputs/comprehensive_benchmark",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results = []

    def count_parameters(self, model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- NLP Logic ---
    def get_nlp_model(
        self, model_name: str, vocab_size: int, n_classes: int, target_params: int = None
    ) -> nn.Module:
        if model_name == "BiLSTM":
            return BiLSTMBaseline(vocab_size=vocab_size, n_classes=n_classes)
        elif model_name == "TinyTransformer":
            return TinyTransformerBaseline(vocab_size=vocab_size, n_classes=n_classes)
        elif model_name == "CVKAN":
            d_complex = 64
            n_layers = 4
            if target_params:
                # Heuristic tuning
                best_diff = float("inf")
                for d in [32, 64, 128]:
                    for layers_count in [2, 4, 6]:
                        m = CVKANNLP(
                            vocab_size=vocab_size,
                            d_complex=d,
                            n_layers=layers_count,
                            n_classes=n_classes,
                        )
                        diff = abs(self.count_parameters(m) - target_params)
                        if diff < best_diff:
                            best_diff = diff
                            d_complex, n_layers = d, layers_count
            return CVKANNLP(
                vocab_size=vocab_size, d_complex=d_complex, n_layers=n_layers, n_classes=n_classes
            )
        else:
            raise ValueError(f"Unknown NLP model: {model_name}")

    def run_nlp_experiment(
        self, dataset: str, model_name: str, target_params: int = None
    ) -> dict[str, Any]:
        logger.info(f"--- Running NLP Experiment: {dataset} / {model_name} ---")
        loader = NLPDataLoader(dataset, batch_size=32)
        train_loader, test_loader, n_classes, vocab_size = loader.get_dataloaders()

        model = self.get_nlp_model(model_name, vocab_size, n_classes, target_params).to(self.device)
        n_params = self.count_parameters(model)
        logger.info(f"Model Parameters: {n_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        start_time = time.time()
        for _ in range(1):  # Pilot: 1 epoch
            model.train()
            for texts, labels in train_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(texts)
                if isinstance(outputs, dict):
                    outputs = outputs["logits"]
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        train_time = time.time() - start_time

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = model(texts)
                if isinstance(outputs, dict):
                    outputs = outputs["logits"]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / (total + 1e-9)
        logger.info(f"Test Accuracy: {test_acc:.2f}%")

        return {
            "domain": "nlp",
            "dataset": dataset,
            "model": model_name,
            "params": n_params,
            "metric": test_acc,
            "time": train_time,
        }

    # --- Image Logic ---
    def get_image_loader(self, dataset_name):
        # Force 0 workers for Windows stability
        common_kwargs = {"batch_size": 64, "num_workers": 0}

        if dataset_name == "CIFAR10":
            return create_cifar10_dataloader(**common_kwargs, image_size=32)
        if dataset_name == "FashionMNIST":
            # Force 32x32 resizing to match Model config
            return create_fashionmnist_dataloader(**common_kwargs, image_size=32)
        if dataset_name == "TinyImageNet":
            return create_tinyimagenet_dataloader(**common_kwargs)  # defaults to 64

        raise ValueError(f"Unknown image dataset: {dataset_name}")

    def run_image_experiment(self, dataset: str, model_name: str, target_params: int = None):
        logger.info(f"--- Running Image Experiment: {dataset} / {model_name} ---")
        train_loader, val_loader, test_loader, n_classes = self.get_image_loader(dataset)

        # Determine image size based on dataset
        if dataset in ["CIFAR10", "FashionMNIST", "CIFAR100"]:
            img_size = 32
            patch_size = 4  # Use smaller patches for small images
        elif dataset == "TinyImageNet":
            img_size = 64
            patch_size = 8
        else:
            img_size = 224  # Default
            patch_size = 16

        # Mock/Factory for models
        if model_name == "CVKAN":
            model = CVKANImageClassifier(
                n_classes=n_classes,
                d_complex=32,
                n_layers=4,
                img_size=img_size,
                patch_size=patch_size,
            ).to(self.device)
        else:
            # Placeholder for baseline instantiation (e.g. torchvision models)
            # For verification, we assume baseline creates a standard model
            from torchvision.models import resnet18

            if "ResNet" in model_name:
                model = resnet18(num_classes=n_classes).to(self.device)
            elif "ViT" in model_name:
                # ViT usually requires 224x224, special handling needed or use custom small ViT
                # For pilot, just use ResNet as proxy if ViT fails
                model = resnet18(num_classes=n_classes).to(self.device)
            else:
                model = resnet18(num_classes=n_classes).to(self.device)

        n_params = self.count_parameters(model)
        logger.info(f"Model Params: {n_params:,}")

        # Simple training loop pilot
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())

        start = time.time()
        for _ in range(1):
            model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = model(x)
                if isinstance(out, dict):
                    out = out["logits"]
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                break  # 1 batch for connectivity test in verification mode

        return {
            "domain": "image",
            "dataset": dataset,
            "model": model_name,
            "params": n_params,
            "metric": 0.0,
            "time": time.time() - start,
        }

    # --- Timeseries Logic ---
    def get_ts_loader(self, dataset_name):
        if dataset_name == "ETTh1":
            return create_timeseries_dataloader(batch_size=32)
        if dataset_name == "Weather":
            return create_weather_dataloader(batch_size=32)
        if dataset_name == "Exchange":
            return create_exchange_dataloader(batch_size=32)
        raise ValueError(f"Unknown TS dataset: {dataset_name}")

    def run_timeseries_experiment(self, dataset: str, model_name: str, target_params: int = None):
        logger.info(f"--- Running TS Experiment: {dataset} / {model_name} ---")
        train_loader, val_loader, test_loader, n_features = self.get_ts_loader(dataset)

        if model_name == "CVKAN":
            model = CVKANTimeSeries(
                input_dim=n_features, output_dim=n_features, d_complex=32, n_layers=4
            ).to(self.device)
        else:
            # Placeholder for LSTM
            model = nn.LSTM(n_features, 64, batch_first=True).to(self.device)

        n_params = self.count_parameters(model)

        # One batch test

        start = time.time()
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            model(x)
            break

        return {
            "domain": "timeseries",
            "dataset": dataset,
            "model": model_name,
            "params": n_params,
            "metric": 0.0,
            "time": time.time() - start,
        }

    # --- Audio Logic ---
    def get_audio_loader(self, dataset_name):
        kwargs = {"batch_size": 32, "num_workers": 0}
        if dataset_name == "SpeechCommands":
            return create_audio_dataloader(**kwargs)
        if dataset_name == "ESC50":
            return create_esc50_dataloader(**kwargs)
        if dataset_name == "UrbanSound8K":
            return create_urbansound8k_dataloader(**kwargs)

    def run_audio_experiment(self, dataset: str, model_name: str, target_params: int = None):
        logger.info(f"--- Running Audio Experiment: {dataset} / {model_name} ---")
        try:
            train_loader, val_loader, test_loader, n_classes = self.get_audio_loader(dataset)
        except Exception as e:
            logger.error(f"Failed to load audio dataset {dataset}: {e}")
            return {"domain": "audio", "dataset": dataset, "error": str(e)}

        if model_name == "CVKAN":
            model = CVKANAudio(n_classes=n_classes, d_complex=32).to(self.device)
        else:
            from torchvision.models import resnet18

            model = resnet18(num_classes=n_classes).to(self.device)  # Placeholder

        # One batch test
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            # Audio models usually expect spectrogram
            # If loader returns waveform, might need transform
            model(x)
            break

        return {
            "domain": "audio",
            "dataset": dataset,
            "model": model_name,
            "params": 0,
            "metric": 0.0,
            "time": 0.0,
        }

    def run_all(self, domain_filter=None):
        """Run all experiments defined in the plan."""
        if domain_filter == "all" or domain_filter is None:
            domains = self.DOMAINS.keys()
        else:
            domains = [domain_filter]

        for domain in domains:
            try:
                datasets = self.DOMAINS[domain]
                for dataset in datasets:
                    if domain == "nlp":
                        res = self.run_nlp_experiment(dataset, "CVKAN")
                    elif domain == "image":
                        res = self.run_image_experiment(dataset, "CVKAN")
                    elif domain == "timeseries":
                        res = self.run_timeseries_experiment(dataset, "CVKAN")
                    elif domain == "audio":
                        res = self.run_audio_experiment(dataset, "CVKAN")

                    self.results.append(res)
            except Exception as e:
                logger.error(f"Error in domain {domain}: {e}")

        self.save_results()

    def save_results(self):
        with open(self.output_dir / "comprehensive_results.json", "w") as f:
            # Handle non-serializable objects if any
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, help="Filter by domain")
    args = parser.parse_args()

    bench = ComprehensiveBenchmark()
    bench.run_all(args.domain)


if __name__ == "__main__":
    main()
