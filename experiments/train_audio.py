"""
Training script for CV-KAN Audio Classification on Speech Commands.

Usage:
    python experiments/train_audio.py --epochs 30 --d_complex 128 --patience 10 --amp
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

from src.data import create_audio_dataloader, TORCHAUDIO_AVAILABLE
from src.models.cv_kan_audio import CVKANAudio
from src.trainer import BaseTrainer


class AudioTrainer(BaseTrainer):
    def train_step(self, batch):
        waveforms, labels = batch
        # waveforms: (batch, 1, samples)
        waveforms = waveforms.squeeze(1).to(self.device)  # (batch, samples)
        labels = labels.to(self.device)
        
        outputs = self.model(waveforms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        
        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }

    def validate_step(self, batch):
        waveforms, labels = batch
        waveforms = waveforms.squeeze(1).to(self.device)
        labels = labels.to(self.device)
        
        outputs = self.model(waveforms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        
        _, predicted = logits.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on Speech Commands')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/speech_commands')
    parser.add_argument('--batch_size', type=int, default=256)  # Larger batch for GPU utilization
    parser.add_argument('--subset_size', type=int, default=None, help='Use subset for pilot runs')
    
    # Model args
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--hop_length', type=int, default=128)
    parser.add_argument('--d_complex', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'attention'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # New args
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/audio')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.metric_mode = 'max' # For accuracy, higher is better
    return args


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    
    if not TORCHAUDIO_AVAILABLE:
        print("ERROR: torchaudio is required for audio benchmarking but is not available.")
        return None
    
    set_seed(args.seed)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"cvkan_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    print("Loading Speech Commands...")
    try:
        train_loader, val_loader, test_loader, n_classes = create_audio_dataloader(
            root=args.data_root,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
            download=True,
        )
    except Exception as e:
        print(f"Failed to load Speech Commands dataset: {e}")
        print("You may need to download the dataset manually or check your internet connection.")
        return None
    
    print(f"Number of classes: {n_classes}, Train batches: {len(train_loader)}")
    
    # Create model
    print("Creating model...")
    model = CVKANAudio(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=n_classes,
        kan_hidden=args.kan_hidden,
        task='classification',
        pooling=args.pooling,
        use_stft_frontend=True,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Trainer
    trainer = AudioTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        args=args,
        use_amp=args.amp
    )
    
    # Training
    history, total_time = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        metric_name='accuracy'
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    # Save final results
    results = {
        'model': 'CVKANAudio',
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
    print(f"Best val acc: {trainer.best_val_metric:.2f}%, Test acc: {test_results['accuracy']:.2f}%")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
