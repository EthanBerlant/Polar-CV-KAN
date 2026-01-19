"""
M5 CNN Baseline for Speech Commands Audio Classification.
Based on "Very Deep Convolutional Neural Networks for Raw Waveforms" (Dai et al., 2017).
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data import create_audio_dataloader, TORCHAUDIO_AVAILABLE

# M5 architecture from https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        # x: (batch, input_len) -> (batch, 1, input_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        
        # Global Average Pooling
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2).squeeze(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Train M5 CNN on Speech Commands')
    
    # Data args
    parser.add_argument('--data_root', type=str, default='./data/speech_commands')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subset_size', type=int, default=None)
    
    # Model args
    parser.add_argument('--d_complex', type=int, default=32, help='Used as channel width multiplier')
    parser.add_argument('--n_layers', type=int, default=4, help='Ignored for M5 (fixed depth)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-2) # M5 usually needs higher LR
    parser.add_argument('--weight_decay', type=float, default=0.0001) # M5 uses reduced weight decay
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/baselines/audio')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--save_every', type=int, default=10)
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
        # waveforms: (batch, 1, samples) or (batch, samples)
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(waveforms)
        
        loss = F.nll_loss(outputs, labels)
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
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        
        outputs = model(waveforms)
        loss = F.nll_loss(outputs, labels)
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
    
    if not TORCHAUDIO_AVAILABLE:
        print("ERROR: torchaudio is required.")
        return None
        
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    run_name = args.run_name or f"m5_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("Loading Speech Commands...")
    # Use our patched loader
    from src.data.audio_data import create_audio_dataloader
    
    try:
        train_loader, val_loader, test_loader, n_classes = create_audio_dataloader(
            root=args.data_root,
            batch_size=args.batch_size,
            subset_size=args.subset_size,
        )
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
        
    print(f"Creating M5 model (channels={args.d_complex})...")
    model = M5(n_input=1, n_output=n_classes, n_channel=args.d_complex).to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
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
              f"Train Acc: {train_results['accuracy']:.2f}% - "
              f"Val Acc: {val_results['accuracy']:.2f}%")
        
        history.append({
            'epoch': epoch,
            'train': train_results,
            'val': val_results,
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
        'model': 'M5-CNN',
        'dataset': 'SpeechCommands',
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
