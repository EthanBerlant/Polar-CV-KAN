"""
Overnight Benchmark Runner for CV-KAN Domain Adaptation.

Runs all three domains (Image, Timeseries, Audio) with parameter sweeps
and multiple repetitions for statistical significance.

Usage:
    python experiments/run_benchmark.py --pilot      # Quick test run
    python experiments/run_benchmark.py --full       # Full overnight run
    python experiments/run_benchmark.py --resume     # Resume from checkpoint
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    domain: str           # 'image', 'timeseries', 'audio'
    model_type: str       # 'cvkan' or baseline name
    d_complex: int
    n_layers: int
    seed: int
    epochs: int
    patience: int = 10
    amp: bool = False
    subset_size: Optional[int] = None
    extra_args: Optional[Dict[str, Any]] = None
    
    @property
    def run_name(self) -> str:
        return f"{self.domain}_{self.model_type}_d{self.d_complex}_L{self.n_layers}_s{self.seed}"


@dataclass 
class ExperimentResult:
    """Result from a single experiment."""
    config: Dict[str, Any]
    success: bool
    metrics: Dict[str, float]
    n_params: int
    train_time_seconds: float
    error_message: Optional[str] = None


class BenchmarkRunner:
    """Orchestrates running all benchmark experiments."""
    
    # Parameter sweep configurations - sized to match baseline models (~200-500k params)
    D_COMPLEX_VALUES = [128, 256]
    N_LAYERS_VALUES = [4, 6]
    SEEDS = [42, 123, 456]
    
    # Domain-specific settings
    DOMAIN_CONFIGS = {
        'image': {
            'epochs': 50,
            'patience': 10,
            'script': 'experiments/train_image.py',
            'baseline_script': 'experiments/baselines/vit_baseline.py',
            'pilot_epochs': 5,
            'pilot_subset': 1000,
            'metrics': ['test_acc', 'best_val_acc'],
        },
        'timeseries': {
            'epochs': 30,
            'patience': 10,
            'script': 'experiments/train_timeseries.py', 
            'baseline_script': 'experiments/baselines/lstm_baseline.py',
            'pilot_epochs': 3,
            'pilot_subset': None,
            'metrics': ['test_mse', 'test_mae'],
        },
        'audio': {
            'epochs': 20,
            'patience': 10,
            'script': 'experiments/train_audio.py',
            'baseline_script': 'experiments/baselines/cnn_audio_baseline.py',
            'pilot_epochs': 2,
            'pilot_subset': 500,
            'metrics': ['test_acc', 'best_val_acc'],
        },
    }
    
    def __init__(self, output_dir: str = 'outputs/benchmark', pilot: bool = False, amp: bool = True, patience: Optional[int] = None, epochs: Optional[int] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pilot = pilot
        self.amp = amp
        self.patience_override = patience
        self.epochs_override = epochs
        self.results: List[ExperimentResult] = []
        self.checkpoint_path = self.output_dir / 'checkpoint.json'
        self.completed_runs: set = set()
        
        # Timestamp for this benchmark session
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def generate_configs(self, domains: Optional[List[str]] = None) -> List[ExperimentConfig]:
        """Generate all experiment configurations.
        
        Interleaves domains horizontally (round-robin) rather than running
        all experiments for one domain before moving to the next.
        """
        domains = domains or ['timeseries', 'image', 'audio']
        model_types = ['cvkan', 'baseline']
        
        seeds = [42] if self.pilot else self.SEEDS
        d_values = [64] if self.pilot else self.D_COMPLEX_VALUES
        n_values = [4] if self.pilot else self.N_LAYERS_VALUES
        
        # Build configs grouped by hyperparameter combo for horizontal slicing
        # This ensures we run one experiment per domain before repeating
        configs = []
        
        for d_complex in d_values:
            for n_layers in n_values:
                for seed in seeds:
                    for model_type in model_types:
                        for domain in domains:
                            domain_cfg = self.DOMAIN_CONFIGS[domain]
                            epochs = self.epochs_override if self.epochs_override else (domain_cfg['pilot_epochs'] if self.pilot else domain_cfg['epochs'])
                            patience = self.patience_override if self.patience_override else domain_cfg['patience']
                            subset = domain_cfg['pilot_subset'] if self.pilot else None
                            
                            configs.append(ExperimentConfig(
                                domain=domain,
                                model_type=model_type,
                                d_complex=d_complex,
                                n_layers=n_layers,
                                seed=seed,
                                epochs=epochs,
                                patience=patience,
                                amp=self.amp,
                                subset_size=subset,
                            ))
        
        return configs
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment via subprocess."""
        domain_cfg = self.DOMAIN_CONFIGS[config.domain]
        
        if config.model_type == 'baseline':
            script = domain_cfg['baseline_script']
        else:
            script = domain_cfg['script']
        
        # Build command
        cmd = [
            sys.executable, script,
            '--epochs', str(config.epochs),
            '--patience', str(config.patience),
            '--d_complex', str(config.d_complex),
            '--n_layers', str(config.n_layers),
            '--seed', str(config.seed),
            '--run_name', config.run_name,
            '--output_dir', str(self.output_dir / config.domain),
        ]
        
        if config.amp:
            cmd.append('--amp')
        
        if config.subset_size:
            cmd.extend(['--subset_size', str(config.subset_size)])
        
        print(f"\n{'='*60}")
        print(f"Running: {config.run_name}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            # Run the training script
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per experiment
            )
            
            train_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"FAILED: {result.stderr[:500]}")
                return ExperimentResult(
                    config=asdict(config),
                    success=False,
                    metrics={},
                    n_params=0,
                    train_time_seconds=train_time,
                    error_message=result.stderr[:1000],
                )
            
            # Load results from JSON
            results_path = self.output_dir / config.domain / config.run_name / 'results.json'
            if results_path.exists():
                with open(results_path) as f:
                    run_results = json.load(f)
                
                metrics = {k: run_results.get(k, 0) for k in domain_cfg['metrics']}
                n_params = run_results.get('n_params', 0)
            else:
                metrics = {}
                n_params = 0
            
            print(f"SUCCESS: {metrics} ({train_time:.1f}s)")
            
            return ExperimentResult(
                config=asdict(config),
                success=True,
                metrics=metrics,
                n_params=n_params,
                train_time_seconds=train_time,
            )
            
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                config=asdict(config),
                success=False,
                metrics={},
                n_params=0,
                train_time_seconds=time.time() - start_time,
                error_message="Experiment timed out after 2 hours",
            )
        except Exception as e:
            return ExperimentResult(
                config=asdict(config),
                success=False,
                metrics={},
                n_params=0,
                train_time_seconds=time.time() - start_time,
                error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )
    
    def save_checkpoint(self):
        """Save progress for crash recovery."""
        checkpoint = {
            'session_id': self.session_id,
            'completed_runs': list(self.completed_runs),
            'results': [asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in self.results],
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> bool:
        """Load checkpoint if exists. Returns True if loaded."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                checkpoint = json.load(f)
            self.completed_runs = set(checkpoint.get('completed_runs', []))
            self.results = checkpoint.get('results', [])
            self.session_id = checkpoint.get('session_id', self.session_id)
            print(f"Resumed from checkpoint: {len(self.completed_runs)} experiments completed")
            return True
        return False
    
    def run_all(self, domains: Optional[List[str]] = None, resume: bool = False):
        """Run all benchmark experiments."""
        if resume:
            self.load_checkpoint()
        
        configs = self.generate_configs(domains)
        total = len(configs)
        skipped = 0
        
        print(f"\n{'#'*60}")
        print(f"# CV-KAN Benchmark Runner")
        print(f"# Mode: {'PILOT' if self.pilot else 'FULL'}")
        print(f"# Total experiments: {total}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        for i, config in enumerate(configs, 1):
            if config.run_name in self.completed_runs:
                print(f"[{i}/{total}] Skipping (already completed): {config.run_name}")
                skipped += 1
                continue
            
            print(f"\n[{i}/{total}] Starting: {config.run_name}")
            
            result = self.run_experiment(config)
            self.results.append(result)
            self.completed_runs.add(config.run_name)
            
            # Save checkpoint after each experiment
            self.save_checkpoint()
        
        total_time = time.time() - start_time
        
        print(f"\n{'#'*60}")
        print(f"# Benchmark Complete!")
        print(f"# Total time: {total_time/3600:.2f} hours")
        print(f"# Experiments run: {total - skipped}")
        print(f"# Skipped (resumed): {skipped}")
        print(f"{'#'*60}\n")
        
        # Generate final report
        self.generate_report()
    
    def generate_report(self):
        """Generate markdown report with summary tables."""
        report_lines = [
            "# CV-KAN Benchmark Results",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nMode: {'Pilot' if self.pilot else 'Full'}",
            "",
        ]
        
        # Group results by domain
        by_domain = {}
        for r in self.results:
            if isinstance(r, dict):
                domain = r['config']['domain']
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(r)
            else:
                domain = r.config['domain']
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(asdict(r))
        
        for domain, results in by_domain.items():
            report_lines.append(f"\n## {domain.title()}\n")
            
            # Create summary table
            metrics = self.DOMAIN_CONFIGS[domain]['metrics']
            header = "| d_complex | n_layers | " + " | ".join(metrics) + " | Params | Status |"
            sep = "|" + "---|" * (4 + len(metrics))
            
            report_lines.append(header)
            report_lines.append(sep)
            
            for r in results:
                cfg = r['config']
                status = "OK" if r['success'] else "FAIL"
                metric_vals = [f"{r['metrics'].get(m, 0):.4f}" for m in metrics]
                params = f"{r['n_params']:,}" if r['n_params'] else "N/A"
                
                row = f"| {cfg['d_complex']} | {cfg['n_layers']} | " + " | ".join(metric_vals) + f" | {params} | {status} |"
                report_lines.append(row)
        
        # Summary statistics
        successful = [r for r in self.results if (r['success'] if isinstance(r, dict) else r.success)]
        failed = len(self.results) - len(successful)
        
        report_lines.extend([
            "\n## Summary",
            f"- Total experiments: {len(self.results)}",
            f"- Successful: {len(successful)}",
            f"- Failed: {failed}",
        ])
        
        report_path = self.output_dir / 'benchmark_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Report saved to: {report_path}")
        
        # Also save raw results as JSON
        results_path = self.output_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'pilot': self.pilot,
                'results': [r if isinstance(r, dict) else asdict(r) for r in self.results],
            }, f, indent=2)
        
        print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='CV-KAN Benchmark Runner')
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--pilot', action='store_true', help='Quick pilot run')
    mode_group.add_argument('--full', action='store_true', help='Full overnight run')
    
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='outputs/benchmark')
    parser.add_argument('--domains', type=str, nargs='+', 
                        choices=['image', 'timeseries', 'audio'],
                        help='Specific domains to run (default: all)')
    parser.add_argument('--no-amp', action='store_true', dest='no_amp', help='Disable Automatic Mixed Precision')
    parser.add_argument('--patience', type=int, default=None, help='Override early stopping patience for all experiments')
    parser.add_argument('--epochs', type=int, default=None, help='Override number of epochs for all experiments')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        pilot=args.pilot,
        amp=not args.no_amp,  # AMP enabled by default
        patience=args.patience,
        epochs=args.epochs,
    )
    
    runner.run_all(domains=args.domains, resume=args.resume)


if __name__ == '__main__':
    main()
