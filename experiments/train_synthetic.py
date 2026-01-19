"""
Training script for CV-KAN on the synthetic signal/noise task.

Usage:
    python experiments/train_synthetic.py --epochs 100 --d_complex 32
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import SignalNoiseDataset, create_signal_noise_dataloader
from src.models import CVKAN
from src.models.cv_kan import CVKANTokenClassifier
from src.losses import diversity_loss, phase_anchor_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train CV-KAN on synthetic task')
    
    # Data args
    parser.add_argument('--n_samples', type=int, default=10000, help='Training samples')
    parser.add_argument('--n_tokens', type=int, default=16, help='Tokens per sequence')
    parser.add_argument('--k_signal', type=int, default=4, help='Signal tokens per sequence')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    # Model args
    parser.add_argument('--d_complex', type=int, default=32, help='Complex dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--kan_hidden', type=int, default=32, help='KAN hidden size')
    parser.add_argument('--head_approach', type=str, default='emergent',
                        choices=['emergent', 'offset', 'factored'])
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads (for offset/factored)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    
    # Regularization args
    parser.add_argument('--diversity_weight', type=float, default=0.1, help='Diversity loss weight')
    parser.add_argument('--anchor_weight', type=float, default=0.01, help='Phase anchor loss weight')
    
    # Task type
    parser.add_argument('--task', type=str, default='token',
                        choices=['token', 'sequence'], help='Classification task type')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def compute_metrics(outputs: dict, labels: torch.Tensor, task: str) -> dict:
    """Compute task-specific metrics."""
    metrics = {}
    
    if task == 'token':
        # Token-level accuracy
        token_logits = outputs['token_logits']  # (batch, n_tokens, 2)
        token_preds = token_logits.argmax(dim=-1)  # (batch, n_tokens)
        token_acc = (token_preds == labels).float().mean()
        metrics['token_acc'] = token_acc.item()
        
        # Signal recall (what fraction of signal tokens correctly identified)
        signal_mask = labels == 1
        if signal_mask.sum() > 0:
            signal_recall = (token_preds[signal_mask] == 1).float().mean()
            metrics['signal_recall'] = signal_recall.item()
        
        # Noise precision (what fraction of predicted noise is actually noise)
        noise_preds = token_preds == 0
        if noise_preds.sum() > 0:
            noise_precision = (labels[noise_preds] == 0).float().mean()
            metrics['noise_precision'] = noise_precision.item()
    else:
        # Sequence-level accuracy
        logits = outputs['logits']  # (batch, 2)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        metrics['acc'] = acc.item()
    
    return metrics


def compute_diagnostics(outputs: dict) -> dict:
    """Compute polarization diagnostics."""
    diagnostics = {}
    
    # Get final complex representation
    if 'Z' in outputs:
        Z = outputs['Z']
    elif 'intermediates' in outputs and outputs['intermediates']:
        Z = outputs['intermediates'][-1]
    else:
        return diagnostics
    
    # Phase coherence across tokens
    phases = torch.angle(Z)
    cos_sum = torch.cos(phases).sum(dim=1)
    sin_sum = torch.sin(phases).sum(dim=1)
    n = phases.shape[1]
    coherence = torch.sqrt(cos_sum ** 2 + sin_sum ** 2) / n
    diagnostics['mean_coherence'] = coherence.mean().item()
    
    # Magnitude statistics
    mags = torch.abs(Z)
    diagnostics['mean_mag'] = mags.mean().item()
    diagnostics['std_mag'] = mags.std().item()
    diagnostics['mag_range'] = (mags.max() - mags.min()).item()
    
    return diagnostics


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    args,
    device: str,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_task_loss = 0
    total_div_loss = 0
    total_anchor_loss = 0
    all_metrics = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        sequences = batch['sequence'].to(device)
        
        if args.task == 'token':
            labels = batch['token_labels'].long().to(device)
        else:
            labels = batch['sequence_label'].long().to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences, return_intermediates=True)
        
        # Task loss
        if args.task == 'token':
            token_logits = outputs['token_logits']  # (batch, n_tokens, 2)
            task_loss = F.cross_entropy(
                token_logits.view(-1, 2),
                labels.view(-1)
            )
        else:
            logits = outputs['logits']
            task_loss = F.cross_entropy(logits, labels)
        
        # Regularization losses
        Z_final = outputs.get('Z', outputs['intermediates'][-1] if 'intermediates' in outputs else None)
        
        div_loss = torch.tensor(0.0, device=device)
        anchor_loss = torch.tensor(0.0, device=device)
        
        if Z_final is not None and args.diversity_weight > 0:
            div_loss = diversity_loss(Z_final)
        
        if Z_final is not None and args.anchor_weight > 0:
            anchor_loss = phase_anchor_loss(Z_final)
        
        # Total loss
        loss = task_loss + args.diversity_weight * div_loss + args.anchor_weight * anchor_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Accumulate
        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_div_loss += div_loss.item()
        total_anchor_loss += anchor_loss.item()
        
        # Metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs, labels, args.task)
            all_metrics.append(metrics)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'task': f'{task_loss.item():.4f}',
        })
    
    n_batches = len(dataloader)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    return {
        'loss': total_loss / n_batches,
        'task_loss': total_task_loss / n_batches,
        'diversity_loss': total_div_loss / n_batches,
        'anchor_loss': total_anchor_loss / n_batches,
        **avg_metrics,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    args,
    device: str,
) -> dict:
    """Evaluate on validation set."""
    model.eval()
    
    total_loss = 0
    all_metrics = []
    all_diagnostics = []
    
    for batch in dataloader:
        sequences = batch['sequence'].to(device)
        
        if args.task == 'token':
            labels = batch['token_labels'].long().to(device)
        else:
            labels = batch['sequence_label'].long().to(device)
        
        outputs = model(sequences, return_intermediates=True)
        
        # Loss
        if args.task == 'token':
            token_logits = outputs['token_logits']
            loss = F.cross_entropy(token_logits.view(-1, 2), labels.view(-1))
        else:
            logits = outputs['logits']
            loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        all_metrics.append(compute_metrics(outputs, labels, args.task))
        all_diagnostics.append(compute_diagnostics(outputs))
    
    n_batches = len(dataloader)
    
    # Average
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m.get(key, 0) for m in all_metrics if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    avg_diagnostics = {}
    for key in all_diagnostics[0].keys():
        values = [d.get(key, 0) for d in all_diagnostics if key in d]
        if values:
            avg_diagnostics[key] = sum(values) / len(values)
    
    return {
        'loss': total_loss / n_batches,
        **avg_metrics,
        **avg_diagnostics,
    }


def main():
    args = parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets
    print("Creating datasets...")
    train_loader = create_signal_noise_dataloader(
        n_samples=args.n_samples,
        n_tokens=args.n_tokens,
        k_signal=args.k_signal,
        d_complex=args.d_complex,
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )
    
    val_loader = create_signal_noise_dataloader(
        n_samples=args.n_samples // 5,
        n_tokens=args.n_tokens,
        k_signal=args.k_signal,
        d_complex=args.d_complex,
        batch_size=args.batch_size,
        shuffle=False,
        seed=123,  # Different seed
    )
    
    # Create model
    print("Creating model...")
    if args.task == 'token':
        model = CVKANTokenClassifier(
            d_input=args.d_complex,
            d_complex=args.d_complex,
            n_layers=args.n_layers,
            n_classes=2,
            kan_hidden=args.kan_hidden,
            head_approach=args.head_approach,
            n_heads=args.n_heads,
            input_type='complex',
        )
    else:
        model = CVKAN(
            d_input=args.d_complex,
            d_complex=args.d_complex,
            n_layers=args.n_layers,
            n_classes=2,
            kan_hidden=args.kan_hidden,
            head_approach=args.head_approach,
            n_heads=args.n_heads,
            input_type='complex',
        )
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("Starting training...")
    best_val_metric = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        
        # Train
        train_results = train_epoch(model, train_loader, optimizer, args, device)
        scheduler.step()
        
        # Evaluate
        val_results = evaluate(model, val_loader, args, device)
        
        # Log
        print(f"Train - Loss: {train_results['loss']:.4f}", end='')
        if args.task == 'token':
            print(f", Token Acc: {train_results.get('token_acc', 0):.4f}")
        else:
            print(f", Acc: {train_results.get('acc', 0):.4f}")
        
        print(f"Val   - Loss: {val_results['loss']:.4f}", end='')
        if args.task == 'token':
            print(f", Token Acc: {val_results.get('token_acc', 0):.4f}", end='')
            print(f", Signal Recall: {val_results.get('signal_recall', 0):.4f}")
        else:
            print(f", Acc: {val_results.get('acc', 0):.4f}")
        
        print(f"Diagnostics - Coherence: {val_results.get('mean_coherence', 0):.4f}, "
              f"Mag Mean: {val_results.get('mean_mag', 0):.4f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': train_results,
            'val': val_results,
            'lr': scheduler.get_last_lr()[0],
        })
        
        # Save best model
        val_metric = val_results.get('token_acc', val_results.get('acc', 0))
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metric': val_metric,
                'args': vars(args),
            }, os.path.join(args.output_dir, 'best.pt'))
            print(f"  -> Saved best model (metric: {val_metric:.4f})")
        
        # Periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'args': vars(args),
            }, os.path.join(args.output_dir, f'checkpoint_{epoch}.pt'))
    
    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'args': vars(args),
    }, os.path.join(args.output_dir, 'final.pt'))
    
    print(f"\nTraining complete! Best validation metric: {best_val_metric:.4f}")


if __name__ == '__main__':
    main()
