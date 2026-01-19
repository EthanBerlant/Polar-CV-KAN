"""
Training script for CV-KAN Audio Classification on Speech Commands.

Usage:
    python experiments/train_audio.py --epochs 30 --d_complex 128
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
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_audio_dataloader, TORCHAUDIO_AVAILABLE
from src.models.cv_kan_audio import CVKANAudio


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on Speech Commands')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/speech_commands')
    parser.add_argument('--batch_size', type=int, default=64)
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
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/audio')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
    for waveforms, labels in pbar:
        # waveforms: (batch, 1, samples)
        waveforms = waveforms.squeeze(1).to(device)  # (batch, samples)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(waveforms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for waveforms, labels in dataloader:
        waveforms = waveforms.squeeze(1).to(device)
        labels = labels.to(device)
        
        outputs = model(waveforms)
        logits = outputs['logits']
        
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


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
    
    # Training loop
    print("Starting training...")
    history = []
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_results = train_epoch(model, train_loader, optimizer, device, epoch)
        val_results = evaluate(model, val_loader, device)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_results['loss']:.4f}, Train Acc: {train_results['accuracy']:.2f}% - "
              f"Val Loss: {val_results['loss']:.4f}, Val Acc: {val_results['accuracy']:.2f}%")
        
        history.append({
            'epoch': epoch,
            'train': train_results,
            'val': val_results,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
        })
        
        # Save best
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_results['accuracy'],
                'args': vars(args),
            }, output_dir / 'best.pt')
            print(f"  -> Saved best model (val acc: {val_results['accuracy']:.2f}%)")
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_{epoch}.pt')
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    total_time = time.time() - start_time
    
    # Save final results
    results = {
        'model': 'CVKANAudio',
        'dataset': 'SpeechCommands',
        'n_params': n_params,
        'best_val_acc': best_val_acc,
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
    print(f"Best val acc: {best_val_acc:.2f}%, Test acc: {test_results['accuracy']:.2f}%")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
