"""
Training script for CV-KAN Time Series Forecasting on ETTh1.

Usage:
    python experiments/train_timeseries.py --epochs 50 --d_complex 64
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

from src.data import create_timeseries_dataloader
from src.models.cv_kan_timeseries import CVKANTimeSeries


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on ETTh1 time series')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/ETT')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=96, help='Lookback window')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon')
    
    # Model args
    parser.add_argument('--d_complex', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--output_mode', type=str, default='real', 
                        choices=['magnitude', 'real', 'phase', 'both'])
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'learnable', 'none'])
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/timeseries')
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


def train_epoch(model, dataloader, optimizer, device, epoch, pred_len):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
    for seq_x, seq_y in pbar:
        seq_x = seq_x.to(device)
        seq_y = seq_y.to(device)
        
        # seq_y contains [label_len + pred_len] timesteps
        # We predict the last pred_len timesteps
        target = seq_y[:, -pred_len:, :]
        
        optimizer.zero_grad()
        
        outputs = model(seq_x, return_sequence=False)
        predictions = outputs['predictions']  # (batch, pred_len, output_dim)
        
        loss = F.mse_loss(predictions, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_mse += F.mse_loss(predictions, target, reduction='sum').item()
        total_mae += F.l1_loss(predictions, target, reduction='sum').item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n_samples = n_batches * dataloader.batch_size * pred_len * target.shape[-1]
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, pred_len):
    model.eval()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    n_batches = 0
    
    for seq_x, seq_y in dataloader:
        seq_x = seq_x.to(device)
        seq_y = seq_y.to(device)
        
        target = seq_y[:, -pred_len:, :]
        
        outputs = model(seq_x, return_sequence=False)
        predictions = outputs['predictions']
        
        loss = F.mse_loss(predictions, target)
        total_loss += loss.item()
        total_mse += F.mse_loss(predictions, target, reduction='sum').item()
        total_mae += F.l1_loss(predictions, target, reduction='sum').item()
        n_batches += 1
    
    n_samples = n_batches * dataloader.batch_size * pred_len * target.shape[-1]
    
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_samples,
        'mae': total_mae / n_samples,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"cvkan_ts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Create dataloaders
    print("Loading ETTh1...")
    train_loader, val_loader, test_loader, input_dim = create_timeseries_dataloader(
        root=args.data_root,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )
    print(f"Input dim: {input_dim}, Train batches: {len(train_loader)}")
    
    # Create model
    print("Creating model...")
    pos_enc = args.pos_encoding if args.pos_encoding != 'none' else None
    
    model = CVKANTimeSeries(
        input_dim=input_dim,
        d_complex=args.d_complex,
        n_layers=args.n_layers,
        output_dim=input_dim,  # Predict all features
        kan_hidden=args.kan_hidden,
        output_mode=args.output_mode,
        forecast_horizon=args.pred_len,
        pos_encoding=pos_enc,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("Starting training...")
    history = []
    best_val_mse = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_results = train_epoch(model, train_loader, optimizer, device, epoch, args.pred_len)
        val_results = evaluate(model, val_loader, device, args.pred_len)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) - "
              f"Train MSE: {train_results['mse']:.6f}, MAE: {train_results['mae']:.6f} - "
              f"Val MSE: {val_results['mse']:.6f}, MAE: {val_results['mae']:.6f}")
        
        history.append({
            'epoch': epoch,
            'train': train_results,
            'val': val_results,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
        })
        
        # Save best
        if val_results['mse'] < best_val_mse:
            best_val_mse = val_results['mse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': val_results['mse'],
                'args': vars(args),
            }, output_dir / 'best.pt')
            print(f"  -> Saved best model (val MSE: {val_results['mse']:.6f})")
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, output_dir / f'checkpoint_{epoch}.pt')
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = evaluate(model, test_loader, device, args.pred_len)
    print(f"Test MSE: {test_results['mse']:.6f}, MAE: {test_results['mae']:.6f}")
    
    total_time = time.time() - start_time
    
    # Save final results
    results = {
        'model': 'CVKANTimeSeries',
        'dataset': 'ETTh1',
        'n_params': n_params,
        'best_val_mse': best_val_mse,
        'test_mse': test_results['mse'],
        'test_mae': test_results['mae'],
        'total_time_seconds': total_time,
        'epochs': args.epochs,
        'history': history,
        'config': vars(args),
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
    print(f"Best val MSE: {best_val_mse:.6f}, Test MSE: {test_results['mse']:.6f}")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
