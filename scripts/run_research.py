#!/usr/bin/env python3
"""Robust Research Runner for Polar CV-KAN

Systematically executes a grid of experiments across domains and strategies.
Features:
- State persistence (research_status.json): Resumes after crashes.
- Subprocess isolation: One experiment per process to prevent memory leaks.
- Error handling: Logs failures but continues to next job.
- Dry run mode: Validate queue without execution.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Constants
STATUS_FILE = "research_status.json"
LOG_DIR = "research_logs"
PYTHON_EXE = sys.executable


@dataclass
class JobConfig:
    domain: str
    preset: str
    run_name: str
    block_type: str = "polarizing"
    aggregation: str = "mean"
    hierarchical_levels: int | None = None
    hierarchical_sharing: str | None = None
    hierarchical_top_down: str | None = None
    subset_size: int | None = None
    epochs: int = 10
    d_complex: int | None = None
    n_layers: int | None = None
    debug: bool = False
    dropout: float | None = None
    batch_norm: bool = False

    def to_cli_args(self) -> list[str]:
        args = [
            PYTHON_EXE,
            "experiments/train.py",
            "--domain",
            self.domain,
            "--preset",
            self.preset,
            "--run_name",
            self.run_name,
            "--block_type",
            self.block_type,
            "--aggregation",
            self.aggregation,
            "--epochs",
            str(self.epochs),
        ]

        if self.subset_size:
            args.extend(["--subset_size", str(self.subset_size)])

        if self.hierarchical_levels:
            args.extend(["--hierarchical_levels", str(self.hierarchical_levels)])
        if self.hierarchical_sharing:
            args.extend(["--hierarchical_sharing", self.hierarchical_sharing])
        if self.hierarchical_top_down:
            args.extend(["--hierarchical_top_down", self.hierarchical_top_down])

        # Capacity overrides
        if self.d_complex:
            args.extend(["--d_complex", str(self.d_complex)])
        if self.n_layers:
            args.extend(["--n_layers", str(self.n_layers)])

        if self.dropout is not None:
            args.extend(["--dropout", str(self.dropout)])

        if self.debug:
            args.append("--debug")

        if self.batch_norm:
            args.append("--batch_norm")

        return args


@dataclass
class JobStatus:
    id: str
    config: dict
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"] = "PENDING"
    start_time: str | None = None
    end_time: str | None = None
    error: str | None = None
    log_file: str | None = None


class ResearchRunner:
    def __init__(self, use_subsets=True):
        self.use_subsets = use_subsets
        self.status_file = Path(STATUS_FILE)
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(exist_ok=True)
        self.jobs: list[JobStatus] = []

    def generate_grid(self):
        """Generate the full experimental grid."""
        domains = [
            {"name": "nlp", "preset": "sst2", "subset": None},  # SST2 is small enough
            {"name": "image", "preset": "cifar10", "subset": 10000},
            {"name": "audio", "preset": "speech_commands_fast", "subset": 10000},
            {"name": "timeseries", "preset": "etth1", "subset": None},  # ETTh1 is small enough
        ]

        strategies = [
            # 1. Capacity Scaling (Base + High Cap)
            # We will use overrides in the loop for high cap, or define it here if we extend the struct?
            # Creating explicit strategy entries for valid combinations.
            # Baseline (for comparison)
            {
                "id": "base",
                "block": "hierarchical",
                "agg": "magnitude_weighted",
                "h_share": "shared",
            },
            # New Poolings
            {"id": "cov", "block": "hierarchical", "agg": "covariance", "h_share": "shared"},
            {"id": "spec", "block": "hierarchical", "agg": "spectral", "h_share": "shared"},
        ]

        # We need a way to inject "high capacity" params.
        # The current JobConfig doesn't have d_complex overrides.
        # I will handle this by creating special separate jobs manually or extending JobConfig.
        # Let's extend JobConfig first? No, easier to just manually build the list.

        jobs = []

        # Domains to focus on
        target_domains = [
            {"name": "image", "preset": "cifar10", "subset": 10000},
            {"name": "audio", "preset": "speech_commands_fast", "subset": 10000},
        ]

        for dom in target_domains:
            # 1. Standard Capacity Runs (Base, Covariance, Spectral)
            for strat in strategies:
                job_id = f"{dom['name']}_{strat['id']}"
                config = JobConfig(
                    domain=dom["name"],
                    preset=dom["preset"],
                    subset_size=dom["subset"],
                    run_name=job_id,
                    block_type=strat["block"],
                    aggregation=strat["agg"],
                    hierarchical_sharing=strat["h_share"],
                    epochs=10,
                )
                jobs.append(JobStatus(id=job_id, config=asdict(config)))

        strategies = [
            # Phase 4: Regularization & Convergence
        ]

        strategies = [
            # Phase 5: SOTA Scale
        ]

        jobs = []

        # 1. SOTA Run
        # d=128, L=6, dropout=0.2, epochs=100, full dataset, batch_norm=True
        # Note: subset_size=None means full dataset
        sota_conf = JobConfig(
            domain="image",
            preset="cifar10",
            subset_size=None,
            run_name="image_sota",
            block_type="hierarchical",
            aggregation="magnitude_weighted",
            hierarchical_sharing="shared",
            epochs=100,
            d_complex=128,
            n_layers=6,
            dropout=0.2,
            batch_norm=True,
        )
        # 2. SOTA Run (Local Window)
        # d=128, L=6, dropout=0.2, epochs=100, full dataset, batch_norm=True, aggregation=window
        sota_local = JobConfig(
            domain="image",
            preset="cifar10",
            subset_size=None,
            run_name="image_sota_local",
            block_type="hierarchical",
            aggregation="window",
            hierarchical_sharing="shared",
            epochs=100,
            d_complex=128,
            n_layers=6,
            dropout=0.2,
            batch_norm=True,
        )
        # 3. Scale 3M (d=192, L=8)
        scale_3m = JobConfig(
            domain="image",
            preset="cifar10",
            subset_size=None,
            run_name="image_scale_3m",
            block_type="hierarchical",
            aggregation="window",
            hierarchical_sharing="shared",
            epochs=100,
            d_complex=192,
            n_layers=8,
            dropout=0.2,
            batch_norm=True,
        )
        # jobs.append(JobStatus(id="image_sota", config=asdict(sota_conf))) # Commented out
        # jobs.append(JobStatus(id="image_sota_local", config=asdict(sota_local))) # Commented out
        # 4. Atomic Hierarchy (L=4 Stacked)
        # d=128, L=4, atomic=True (inferred by backbone), magnitude_weighted
        atomic_hier = JobConfig(
            domain="image",
            preset="cifar10",
            subset_size=None,
            run_name="image_hierarchical_atomic",
            block_type="hierarchical",
            aggregation="magnitude_weighted",
            hierarchical_sharing="shared",
            epochs=100,
            d_complex=128,
            n_layers=4,
            dropout=0.2,
            batch_norm=True,
        )
        jobs.append(JobStatus(id="image_hierarchical_atomic", config=asdict(atomic_hier)))

        return jobs

    def load_state(self):
        """Load state from disk or initialize if new."""
        if self.status_file.exists():
            print(f"Loading existing state from {self.status_file}")
            with open(self.status_file) as f:
                data = json.load(f)
                self.jobs = [JobStatus(**j) for j in data]
        else:
            print("Initializing new research campaign")
            self.jobs = self.generate_grid()
            self.save_state()

    def save_state(self):
        """Save current state to disk."""
        data = [asdict(j) for j in self.jobs]
        with open(self.status_file, "w") as f:
            json.dump(data, f, indent=2)

    def run(self, dry_run=False):
        """Execute the jobs."""
        print(f"Found {len(self.jobs)} jobs. Status summary:")
        summary = {}
        for j in self.jobs:
            summary[j.status] = summary.get(j.status, 0) + 1
        print(summary)

        if dry_run:
            print("\nDry Run - Queue:")
            for j in self.jobs:
                if j.status == "PENDING":
                    print(f"- {j.id}: {j.config}")
            return

        for job in self.jobs:
            if job.status == "COMPLETED":
                continue

            # If a job was running when we crashed, mark it failed to be safe (or restart)
            # For this script, we'll just restart PENDING/FAILED/RUNNING

            print(f"\n>>> Starting Job: {job.id}")
            job.status = "RUNNING"
            job.start_time = datetime.datetime.now().isoformat()
            self.save_state()

            # Prepare logging
            log_path = self.log_dir / f"{job.id}.log"
            job.log_file = str(log_path)

            config = JobConfig(**job.config)
            cmd = config.to_cli_args()

            print(f"Command: {' '.join(cmd)}")

            with open(log_path, "w", encoding="utf-8") as log_f:
                try:
                    # Prepare env with forced encoding
                    env = os.environ.copy()
                    env["PYTHONIOENCODING"] = "utf-8"

                    # Run subprocess
                    result = subprocess.run(
                        cmd,
                        cwd=os.getcwd(),
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=True,
                        encoding="utf-8",  # Force read encoding
                    )
                    job.status = "COMPLETED"
                    job.end_time = datetime.datetime.now().isoformat()
                    print(f"✅ Job {job.id} completed successfully.")

                except subprocess.CalledProcessError as e:
                    job.status = "FAILED"
                    job.end_time = datetime.datetime.now().isoformat()
                    job.error = f"Exit code {e.returncode}"
                    print(f"❌ Job {job.id} failed. See logs at {log_path}")
                except Exception as e:
                    job.status = "FAILED"
                    job.end_time = datetime.datetime.now().isoformat()
                    job.error = str(e)
                    print(f"❌ Job {job.id} crashed: {e}")

            self.save_state()

            # Brief pause to let file handles close / user interrupt
            time.sleep(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Delete existing state")
    args = parser.parse_args()

    if args.reset and Path(STATUS_FILE).exists():
        Path(STATUS_FILE).unlink()

    runner = ResearchRunner()
    runner.load_state()
    runner.run(dry_run=args.dry_run)
