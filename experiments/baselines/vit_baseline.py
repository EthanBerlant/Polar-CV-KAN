"""
ViT-Tiny Baseline for CIFAR-10 Image Classification.

Designed to match parameter count of CV-KAN models (~50k-300k parameters).
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_cifar10_dataloader


class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=64, depth=6, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.patch_size = patch_size

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).contiguous().view(img.shape[0], -1, p * p * 3)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ViT Baseline on CIFAR-10')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/cifar10')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--subset_size', type=int, default=None)
    
    # Model args
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--patch_size', type=int, default=4)
    # Match CV-KAN args for runner compatibility where possible, but mapped to ViT params
    parser.add_argument('--d_complex', type=int, default=64, help='Used as embedding dim')
    parser.add_argument('--n_layers', type=int, default=6, help='Transformer depth')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/baselines/image')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=20)
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
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
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
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"vit_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading CIFAR-10...")
    train_loader, val_loader, test_loader, n_classes = create_cifar10_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.img_size,
        subset_size=args.subset_size,
    )
    
    print(f"Creating ViT model (dim={args.d_complex}, depth={args.n_layers})...")
    # Map d_complex to dim, n_layers to depth
    model = ViT(
        image_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=n_classes,
        dim=args.d_complex,
        depth=args.n_layers,
        heads=4,
        mlp_dim=args.d_complex * 2,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1)
    
    history = []
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_results = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_results = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train Acc: {train_results['accuracy']:.2f}% - "
              f"Val Acc: {val_results['accuracy']:.2f}%")
        
        history.append({
            'epoch': epoch,
            'train': train_results,
            'val': val_results,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
        })
        
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'val_acc': best_val_acc,
            }, output_dir / 'best.pt')
    
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    
    results = {
        'model': 'ViT-Tiny',
        'dataset': 'CIFAR-10',
        'n_params': n_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_results['accuracy'],
        'test_loss': test_results['loss'],
        'total_time_seconds': time.time() - start_time,
        'epochs': args.epochs,
        'history': history,
        'config': vars(args),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == '__main__':
    main()
