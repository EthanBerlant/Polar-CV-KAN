"""
Training script for CV-KAN Audio Classification using precomputed spectrograms.

This is faster than train_audio.py because it skips STFT computation.

Usage:
    python experiments/train_audio_precomputed.py --epochs 30 --d_complex 128 --amp
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.precomputed_audio import create_precomputed_audio_dataloader
from src.models.base import BaseCVKAN, build_classifier_head
from src.trainer import BaseTrainer


class CVKANAudioPrecomputed(BaseCVKAN):
    """
    CV-KAN for precomputed spectrograms.
    
    Input is (batch, 2, time, freq) where dim 0,1 are real/imag parts.
    Converts to complex, applies CV-KAN layers, pools, classifies.
    """
    
    def __init__(
        self,
        n_freq: int = 257,  # n_fft=512 -> 257 freq bins
        d_complex: int = 128,
        n_layers: int = 4,
        n_classes: int = 35,
        kan_hidden: int = 32,
        pooling: str = 'mean',
    ):
        super().__init__(
            d_complex=d_complex,
            n_layers=n_layers,
            kan_hidden=kan_hidden,
            pooling=pooling,
            center_magnitudes=True,
        )
        
        self.n_freq = n_freq
        
        # Project freq bins to d_complex if different
        if n_freq != d_complex:
            self.freq_proj = nn.Linear(n_freq, d_complex)
        else:
            self.freq_proj = None
        
        # Classifier
        self.classifier = build_classifier_head(
            d_complex=d_complex,
            n_classes=n_classes,
            hidden_dim=kan_hidden * 2,
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 2, time, freq) - stacked real/imag from precompute
        """
        # Convert to complex: (batch, time, freq)
        z = torch.complex(x[:, 0], x[:, 1])
        
        # Project freq dimension if needed
        if self.freq_proj is not None:
            # Project real and imag separately
            z_real = self.freq_proj(z.real)
            z_imag = self.freq_proj(z.imag)
            z = torch.complex(z_real, z_imag)
        
        # Apply CV-KAN layers
        z = self._apply_layers(z)
        
        # Pool and classify
        pooled = self._pool(z)
        features = self._extract_features(pooled)
        logits = self.classifier(features)
        
        return {'logits': logits, 'features': features}


class PrecomputedAudioTrainer(BaseTrainer):
    def train_step(self, batch):
        spectrograms, labels = batch
        spectrograms = spectrograms.to(self.device)  # (batch, 2, time, freq)
        labels = labels.to(self.device)
        
        outputs = self.model(spectrograms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        
        _, predicted = logits.max(1)
        accuracy = 100. * predicted.eq(labels).sum().item() / labels.size(0)
        
        return {'loss': loss, 'accuracy': accuracy}

    def validate_step(self, batch):
        spectrograms, labels = batch
        spectrograms = spectrograms.to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(spectrograms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        
        _, predicted = logits.max(1)
        accuracy = 100. * predicted.eq(labels).sum().item() / labels.size(0)
        
        return {'loss': loss, 'accuracy': accuracy}


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on Precomputed Speech Commands')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/speech_commands_stft')
    parser.add_argument('--batch_size', type=int, default=256)
    
    # Model args
    parser.add_argument('--n_freq', type=int, default=257)  # n_fft=512 -> 257
    parser.add_argument('--d_complex', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='mean')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/audio')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.metric_mode = 'max'
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"cvkan_audio_precomputed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load precomputed spectrograms
    print("Loading precomputed spectrograms...")
    train_loader, val_loader, test_loader, n_classes = create_precomputed_audio_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
    )
    print(f"Classes: {n_classes}, Train batches: {len(train_loader)}")
    
    # Create model
    print("Creating model...")
    model = CVKANAudioPrecomputed(
        n_freq=args.n_freq,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=n_classes,
        kan_hidden=args.kan_hidden,
        pooling=args.pooling,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = PrecomputedAudioTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        args=args,
        use_amp=args.amp
    )
    
    history, total_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        metric_name='accuracy'
    )
    
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    results = {
        'model': 'CVKANAudioPrecomputed',
        'dataset': 'SpeechCommands',
        'n_params': n_params,
        'best_val_acc': trainer.best_val_metric,
        'test_acc': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'total_time_seconds': total_time,
        'epochs': args.epochs,
        'history': history,
        'config': vars(args),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
