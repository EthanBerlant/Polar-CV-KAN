"""
SST-2 configuration sweep for CV-KAN.

Grid search over:
- Layers: 1, 2, 4
- Dimensions: 32, 64, 128  
- KAN hidden: 16, 32

Usage:
    python experiments/sweep_sst2.py --epochs 5
    python experiments/sweep_sst2.py --epochs 1 --configs 2  # Quick test
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from datetime import datetime
from itertools import product

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.text import load_sst2, pad_collate
from src.models import CVKAN
from src.losses import diversity_loss


class TextClassifier(nn.Module):
    """Wrapper for SST-2 classification."""
    
    def __init__(self, vocab_size, embed_dim, cvkan_args):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)
        
    def forward(self, indices, mask=None, return_intermediates=False):
        x = self.embedding(indices)
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


def train_epoch(model, dataloader, optimizer, device, diversity_weight=0.1):
    model.train()
    total_loss, total_acc = 0, 0
    
    for batch in dataloader:
        indices = batch['indices'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(indices, mask=mask, return_intermediates=True)
        logits = outputs['logits']
        
        task_loss = nn.functional.cross_entropy(logits, labels)
        
        Z_final = outputs.get('Z', outputs['intermediates'][-1] if 'intermediates' in outputs else None)
        reg_loss = 0
        if Z_final is not None and diversity_weight > 0:
            reg_loss = diversity_weight * diversity_loss(Z_final)
        
        loss = task_loss + reg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / len(dataloader), total_acc / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_acc = 0, 0
    
    for batch in dataloader:
        indices = batch['indices'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(indices, mask=mask)
        logits = outputs['logits']
        
        loss = nn.functional.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / len(dataloader), total_acc / len(dataloader)


def run_config(config, train_loader, val_loader, vocab_size, device, args):
    """Train and evaluate a single configuration."""
    n_layers, d_complex, kan_hidden = config
    
    cvkan_args = {
        'd_complex': d_complex,
        'n_layers': n_layers,
        'n_classes': 2,
        'kan_hidden': kan_hidden,
        'head_approach': 'emergent',
        'pooling': 'mean',
        'input_type': 'real',
    }
    
    # Match embed_dim to d_complex for simplicity
    model = TextClassifier(
        vocab_size=vocab_size,
        embed_dim=d_complex,
        cvkan_args=cvkan_args,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    best_val_acc = 0
    best_epoch = 0
    final_train_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            diversity_weight=args.diversity_weight
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        
        final_train_acc = train_acc
    
    return {
        'n_layers': n_layers,
        'd_complex': d_complex,
        'kan_hidden': kan_hidden,
        'params': n_params,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': final_train_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--diversity_weight', type=float, default=0.1)
    parser.add_argument('--configs', type=int, default=None,
                        help='Limit number of configs to test (for debugging)')
    parser.add_argument('--output_dir', type=str, default='outputs/sweep')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading SST-2...")
    train_ds, val_ds, vocab = load_sst2(max_len=64)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Define grid
    layers_options = [1, 2, 4]
    dims_options = [32, 64, 128]
    kan_hidden_options = [16, 32]
    
    all_configs = list(product(layers_options, dims_options, kan_hidden_options))
    
    if args.configs:
        all_configs = all_configs[:args.configs]
    
    print(f"\nRunning {len(all_configs)} configurations...")
    print(f"Grid: layers={layers_options}, dims={dims_options}, kan_hidden={kan_hidden_options}")
    
    # Run sweep
    results = []
    for i, config in enumerate(all_configs):
        n_layers, d_complex, kan_hidden = config
        print(f"\n[{i+1}/{len(all_configs)}] layers={n_layers}, dims={d_complex}, kan={kan_hidden}")
        
        result = run_config(config, train_loader, val_loader, len(vocab), device, args)
        results.append(result)
        
        print(f"  -> Params: {result['params']:,} | Best Val Acc: {result['best_val_acc']:.4f}")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)
    
    # Print summary
    print("\n" + "="*80)
    print("CONFIGURATION SWEEP RESULTS (sorted by validation accuracy)")
    print("="*80)
    print(f"{'Layers':>6} {'Dims':>6} {'KAN':>6} {'Params':>12} {'Val Acc':>10} {'Best Ep':>8}")
    print("-"*80)
    for r in results:
        print(f"{r['n_layers']:>6} {r['d_complex']:>6} {r['kan_hidden']:>6} "
              f"{r['params']:>12,} {r['best_val_acc']:>10.4f} {r['best_epoch']:>8}")
    
    # Save CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(args.output_dir, f'sweep_results_{timestamp}.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_file}")
    
    # Best config
    best = results[0]
    print(f"\nBest configuration:")
    print(f"  Layers: {best['n_layers']}")
    print(f"  Dimensions: {best['d_complex']}")
    print(f"  KAN hidden: {best['kan_hidden']}")
    print(f"  Parameters: {best['params']:,}")
    print(f"  Validation Accuracy: {best['best_val_acc']:.4f}")


if __name__ == '__main__':
    main()
