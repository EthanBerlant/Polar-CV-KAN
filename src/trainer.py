import torch
import torch.nn as nn
from tqdm import tqdm
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

class EarlyStopper:
    """Early stopping to stop training when validation loss stops improving."""
    def __init__(self, patience: int = 10, min_delta: float = 0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_at_best = None

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            self.val_score_at_best = val_score
        elif self._is_better(val_score, self.best_score):
            self.best_score = val_score
            self.val_score_at_best = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == 'min':
            return current < (best - self.min_delta)
        else: # mode == 'max'
            return current > (best + self.min_delta)


class BaseTrainer:
    """Base class for all training scripts."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str,
        output_dir: Path,
        args: Any,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.args = args
        self.use_amp = use_amp
        
        # AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda') if use_amp and torch.cuda.is_available() else None
        
        # Logging
        self.history = []
        self.best_val_metric = float('inf') if args.metric_mode == 'min' else float('-inf')
        
    def train_step(self, batch) -> Dict[str, float]:
        """Implement domain-specific training logic."""
        raise NotImplementedError
        
    def validate_step(self, batch) -> Dict[str, float]:
        """Implement domain-specific validation logic."""
        raise NotImplementedError
        
    def train_epoch(self, dataloader, epoch) -> Dict[str, float]:
        self.model.train()
        total_metrics = {}
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
        for batch in pbar:
            # Move batch to device is handled in train_step usually, 
            # but let's leave it to implementations or do it here if standard.
            # We'll let implementations handle device movement for flexibility.
            
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler:
                with torch.amp.autocast('cuda'):
                    metrics = self.train_step(batch)
                    loss = metrics['loss']
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                metrics = self.train_step(batch)
                loss = metrics['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # Accumulate metrics
            for k, v in metrics.items():
                val_item = v.item() if torch.is_tensor(v) else v
                total_metrics[k] = total_metrics.get(k, 0) + val_item
                
            n_batches += 1
            
            # Update pbar
            # display_metrics = {k: f"{v:.4f}" for k,v in metrics.items()}
            # pbar.set_postfix(display_metrics)
            # Only show loss in progress bar to avoid clutter
            pbar.set_postfix({'loss': f"{metrics['loss'].item():.4f}"})
            
        # Average metrics
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        return avg_metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {}
        n_batches = 0
        
        for batch in dataloader:
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    metrics = self.validate_step(batch)
            else:
                metrics = self.validate_step(batch)
            
            for k, v in metrics.items():
                val_item = v.item() if torch.is_tensor(v) else v
                total_metrics[k] = total_metrics.get(k, 0) + val_item
            n_batches += 1
            
        avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
        return avg_metrics

    def fit(self, train_loader, val_loader, epochs: int, patience: int = 10, metric_name: str = 'accuracy'):
        early_stopper = EarlyStopper(patience=patience, mode=self.args.metric_mode)
        start_time = time.time()
        
        print(f"Starting training with patience={patience}, amp={self.use_amp}")
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            
            if self.scheduler:
                # Handle different scheduler types
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[metric_name])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Print status
            train_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            val_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - Train [{train_str}] - Val [{val_str}]")
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': current_lr,
                'epoch_time': epoch_time,
            })
            
            # Checkpoint best
            val_score = val_metrics[metric_name]
            is_best = False
            if self.args.metric_mode == 'min':
                if val_score < self.best_val_metric:
                    is_best = True
            else:
                if val_score > self.best_val_metric:
                    is_best = True
            
            if is_best:
                self.best_val_metric = val_score
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': vars(self.args),
                }, self.output_dir / 'best.pt')
                print(f"  -> Saved best model ({metric_name}: {val_score:.4f})")
            
            # Periodic save
            if epoch % self.args.save_every == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'history': self.history,
                }, self.output_dir / f'checkpoint_{epoch}.pt')
            
            # Early stopping
            if early_stopper(val_score):
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        total_time = time.time() - start_time
        return self.history, total_time
