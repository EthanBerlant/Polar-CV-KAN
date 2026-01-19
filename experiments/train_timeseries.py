"""
Training script for CV-KAN Time Series Forecasting on ETTh1.

Usage:
    python experiments/train_timeseries.py --epochs 50 --d_complex 64 --patience 10 --amp
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

from src.data import create_timeseries_dataloader
from src.models.cv_kan_timeseries import CVKANTimeSeries
from src.trainer import BaseTrainer


class TimeSeriesTrainer(BaseTrainer):
    def train_step(self, batch):
        seq_x, seq_y = batch
        seq_x = seq_x.to(self.device)
        seq_y = seq_y.to(self.device)
        
        # seq_y contains [label_len + pred_len] timesteps
        # We predict the last pred_len timesteps
        pred_len = self.args.pred_len
        target = seq_y[:, -pred_len:, :]
        
        outputs = self.model(seq_x, return_sequence=False)
        predictions = outputs['predictions']  # (batch, pred_len, output_dim)
        
        loss = F.mse_loss(predictions, target)
        mae = F.l1_loss(predictions, target)
        
        return {
            'loss': loss,
            'mse': loss, # Same as loss for this task
            'mae': mae
        }

    def validate_step(self, batch):
        seq_x, seq_y = batch
        seq_x = seq_x.to(self.device)
        seq_y = seq_y.to(self.device)
        
        pred_len = self.args.pred_len
        target = seq_y[:, -pred_len:, :]
        
        outputs = self.model(seq_x, return_sequence=False)
        predictions = outputs['predictions']
        
        loss = F.mse_loss(predictions, target)
        mae = F.l1_loss(predictions, target)
        
        return {
            'loss': loss,
            'mse': loss,
            'mae': mae
        }


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
    
    # New args
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--amp', action='store_true', help='Use Automatic Mixed Precision')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/timeseries')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=10)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    args.metric_mode = 'min' # For MSE, lower is better
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
    
    # Trainer
    trainer = TimeSeriesTrainer(
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
        metric_name='mse'
    )
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_loader)
    print(f"Test MSE: {test_results['mse']:.6f}, MAE: {test_results['mae']:.6f}")
    
    # Save final results
    results = {
        'model': 'CVKANTimeSeries',
        'dataset': 'ETTh1',
        'n_params': n_params,
        'best_val_mse': trainer.best_val_metric,
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
    print(f"Best val MSE: {trainer.best_val_metric:.6f}, Test MSE: {test_results['mse']:.6f}")
    print(f"Results saved to {output_dir}")
    
    return results


if __name__ == '__main__':
    main()
