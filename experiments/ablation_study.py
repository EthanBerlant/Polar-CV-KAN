"""
Ablation study for CV-KAN components.

Tests the contribution of:
- Polarization mechanism
- Diversity loss
- Complex layer normalization

Usage:
    python experiments/ablation_study.py --epochs 5
    python experiments/ablation_study.py --ablation no_polar --epochs 5
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.text import load_sst2, pad_collate
from src.models import CVKAN
from src.losses import diversity_loss, phase_anchor_loss


class IdentityBlock(nn.Module):
    """Identity block to replace PolarizingBlock in ablation."""
    def __init__(self, d_complex, **kwargs):
        super().__init__()
        self.d_complex = d_complex
    
    def forward(self, Z, mask=None):
        return Z


class TextClassifier(nn.Module):
    """Wrapper for SST-2 classification."""
    
    def __init__(self, vocab_size, embed_dim, cvkan_args, use_identity_block=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)
        
        # Ablation: replace polarizing blocks with identity
        if use_identity_block:
            from src.modules.multi_head import EmergentHeadsPolarizing
            for i, layer in enumerate(self.model.layers):
                if isinstance(layer, EmergentHeadsPolarizing):
                    self.model.layers[i] = IdentityBlock(cvkan_args['d_complex'])
        
    def forward(self, indices, mask=None, return_intermediates=False):
        x = self.embedding(indices)
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


class TextClassifierNoNorm(nn.Module):
    """Wrapper without layer normalization."""
    
    def __init__(self, vocab_size, embed_dim, cvkan_args):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.model = CVKAN(d_input=embed_dim, **cvkan_args)
        
        # Replace norms with identity
        for i in range(len(self.model.norms)):
            self.model.norms[i] = nn.Identity()
        
    def forward(self, indices, mask=None, return_intermediates=False):
        x = self.embedding(indices)
        return self.model(x, mask=mask, return_intermediates=return_intermediates)


def train_epoch(model, dataloader, optimizer, device, diversity_weight=0.1, anchor_weight=0.0):
    model.train()
    total_loss, total_acc = 0, 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        indices = batch['indices'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(indices, mask=mask, return_intermediates=True)
        logits = outputs['logits']
        
        task_loss = nn.functional.cross_entropy(logits, labels)
        
        # Regularization
        Z_final = outputs.get('Z', outputs['intermediates'][-1] if 'intermediates' in outputs else None)
        reg_loss = 0
        if Z_final is not None:
            if diversity_weight > 0:
                reg_loss += diversity_weight * diversity_loss(Z_final)
            if anchor_weight > 0:
                reg_loss += anchor_weight * phase_anchor_loss(Z_final)
        
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


def run_ablation(ablation_name, train_loader, val_loader, vocab_size, device, args):
    """Run a single ablation experiment."""
    print(f"\n{'='*50}")
    print(f"Running ablation: {ablation_name}")
    print(f"{'='*50}")
    
    cvkan_args = {
        'd_complex': args.d_complex,
        'n_layers': args.n_layers,
        'n_classes': 2,
        'kan_hidden': args.kan_hidden,
        'head_approach': 'emergent',
        'pooling': 'mean',
        'input_type': 'real',
    }
    
    # Configure based on ablation
    use_identity = False
    diversity_weight = args.diversity_weight
    
    if ablation_name == 'baseline':
        pass  # Full model
    elif ablation_name == 'no_polar':
        use_identity = True
    elif ablation_name == 'no_diversity':
        diversity_weight = 0.0
    elif ablation_name == 'no_norm':
        model = TextClassifierNoNorm(
            vocab_size=vocab_size,
            embed_dim=args.d_embed,
            cvkan_args=cvkan_args,
        ).to(device)
    
    if ablation_name != 'no_norm':
        model = TextClassifier(
            vocab_size=vocab_size,
            embed_dim=args.d_embed,
            cvkan_args=cvkan_args,
            use_identity_block=use_identity,
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    best_val_acc = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, 
            diversity_weight=diversity_weight
        )
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}: Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return {
        'ablation': ablation_name,
        'params': n_params,
        'best_val_acc': best_val_acc,
        'final_val_acc': val_acc,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_embed', type=int, default=64)
    parser.add_argument('--d_complex', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--kan_hidden', type=int, default=32)
    parser.add_argument('--diversity_weight', type=float, default=0.1)
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all', 'baseline', 'no_polar', 'no_diversity', 'no_norm'])
    parser.add_argument('--output_dir', type=str, default='outputs/ablations')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading SST-2...")
    train_ds, val_ds, vocab = load_sst2(max_len=64)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    
    # Define ablations to run
    if args.ablation == 'all':
        ablations = ['baseline', 'no_polar', 'no_diversity', 'no_norm']
    else:
        ablations = [args.ablation]
    
    # Run ablations
    results = []
    for ablation in ablations:
        result = run_ablation(ablation, train_loader, val_loader, len(vocab), device, args)
        results.append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Ablation':<15} {'Params':>10} {'Best Val Acc':>15}")
    print("-"*60)
    for r in results:
        print(f"{r['ablation']:<15} {r['params']:>10,} {r['best_val_acc']:>15.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output_dir, f'ablation_results_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
