"""
Training script for CV-KAN Image Classification on CIFAR-10.

Usage:
    python experiments/train_image.py --epochs 100 --d_complex 64
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import create_cifar10_dataloader
from src.models.cv_kan_image import CVKANImageClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on CIFAR-10')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--subset_size', type=int, default=None, help='Use subset for pilot runs')
    
    # Model args
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--d_complex', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--aggregation', type=str, default='local', choices=['global', 'local'])
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal', choices=['sinusoidal', 'learnable'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/image')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=20)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
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
    
    scheduler.step()
    
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
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
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
    set_seed(args.seed)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"cvkan_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.img_size,
        subset_size=args.subset_size,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("Creating model...")
    model = CVKANImageClassifier(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=3,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        n_classes=n_classes,
        kan_hidden=args.kan_hidden,
        aggregation=args.aggregation,
        pos_encoding=args.pos_encoding,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)
    
    # Training loop
    print("Starting training...")
    history = []
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_results = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_results = evaluate(model, val_loader, device)
        
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
        'model': 'CVKANImageClassifier',
        'dataset': 'CIFAR-10',
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
