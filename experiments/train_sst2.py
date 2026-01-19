"""
Train CV-KAN on SST-2 Sentiment Analysis.

Usage:
    python experiments/train_sst2.py --epochs 10 --d_complex 64
    python experiments/train_sst2.py --model_type transformer --epochs 10
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.text import load_sst2, pad_collate
from src.models import CVKAN
from src.models.baseline_transformer import BaselineTransformer
from src.losses import diversity_loss, phase_anchor_loss


class TextClassifier(nn.Module):
    """Wrapper that adds embedding layer to CVKAN."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        cvkan_args: dict,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)
        
    def forward(self, indices, mask=None, return_intermediates=False):
        # Embed
        x = self.embedding(indices)
        # Pass to CVKAN
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


def train_epoch(model, dataloader, optimizer, device, args, is_cvkan=True):
    model.train()
    total_loss, total_acc = 0, 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        indices = batch['indices'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(indices, mask=mask, return_intermediates=is_cvkan)
        logits = outputs['logits']
        
        task_loss = nn.functional.cross_entropy(logits, labels)
        
        # Regularization (only for CVKAN)
        reg_loss = 0
        if is_cvkan:
            Z_final = outputs.get('Z', outputs['intermediates'][-1] if 'intermediates' in outputs else None)
            if Z_final is not None:
                 if args.diversity_weight > 0:
                     reg_loss += args.diversity_weight * diversity_loss(Z_final)
                 if args.anchor_weight > 0:
                     reg_loss += args.anchor_weight * phase_anchor_loss(Z_final)
        
        loss = task_loss + reg_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc.item():.4f}'})
        
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # Model selection
    parser.add_argument('--model_type', type=str, default='cvkan',
                        choices=['cvkan', 'transformer'],
                        help='Model type: cvkan or transformer')
    
    # Model config (shared)
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension (d_complex for CVKAN, d_model for Transformer)')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--pooling', type=str, default='mean')
    
    # CVKAN-specific
    parser.add_argument('--d_embed', type=int, default=64)
    parser.add_argument('--d_complex', type=int, default=None,
                        help='Alias for d_model (CVKAN)')
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--head_approach', type=str, default='emergent')
    parser.add_argument('--norm_type', type=str, default='none',
                        choices=['layer', 'rms', 'none'],
                        help='Normalization type: layer, rms, or none')
    parser.add_argument('--block_type', type=str, default='polarizing',
                        choices=['polarizing', 'attention'],
                        help='Block type: polarizing or attention')
    
    # Transformer-specific
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Reg
    parser.add_argument('--diversity_weight', type=float, default=0.1)
    parser.add_argument('--anchor_weight', type=float, default=0.0)
    
    parser.add_argument('--output_dir', type=str, default='outputs/sst2')
    args = parser.parse_args()
    
    # Handle d_complex alias
    if args.d_complex is not None:
        args.d_model = args.d_complex
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Data
    print("Loading SST-2...")
    train_ds, val_ds, vocab = load_sst2(max_len=64)
    print(f"Vocab size: {len(vocab)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Create Model
    is_cvkan = args.model_type == 'cvkan'
    
    if is_cvkan:
        model = TextClassifier(
            vocab_size=len(vocab),
            embed_dim=args.d_embed,
            cvkan_args={
                'd_complex': args.d_model,
                'n_layers': args.n_layers,
                'n_classes': 2,
                'kan_hidden': args.kan_hidden,
                'head_approach': args.head_approach,
                'n_heads': args.n_heads,
                'pooling': args.pooling,
                'input_type': 'real',
                'norm_type': args.norm_type,
                'block_type': args.block_type,
            }
        ).to(device)
    else:
        model = BaselineTransformer(
            vocab_size=len(vocab),
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_classes=2,
            dropout=args.dropout,
            pooling=args.pooling,
        ).to(device)
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, args, is_cvkan)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pt'))
            print(f"-> Saved new best: {best_acc:.4f}")

if __name__ == '__main__':
    main()

