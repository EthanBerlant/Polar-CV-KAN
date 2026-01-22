#!/usr/bin/env python3
"""Architecture validation script for Polar CV-KAN.

Checks for:
1. Duplicate training loops outside sanctioned files
2. Orphaned scripts not using domain adapters
3. Benchmark script proliferation
4. Documentation duplication
5. Shell script wrappers

Run with: python scripts/check_architecture.py
"""

import sys
from pathlib import Path


def check_duplicate_training_loops() -> list[str]:
    """Check for training loop duplications outside allowed files."""
    errors = []
    allowed_files = {
        "base_trainer.py",  # Baseline shared infrastructure
        "trainer.py",  # CV-KAN shared infrastructure
    }

    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return errors

    for py_file in experiments_dir.rglob("*.py"):
        if py_file.name in allowed_files:
            continue

        content = py_file.read_text(encoding="utf-8", errors="ignore")

        # Check for standalone training loop implementations
        patterns_to_check = [
            ("def train_epoch(", "training loop"),
            ("for epoch in range(", "epoch loop"),
        ]

        for pattern, description in patterns_to_check:
            if pattern in content:
                # Allow in domain trainers that inherit from base
                if "class " in content and "(BaseTrainer)" in content:
                    continue
                if "class " in content and "(BaselineTrainer)" in content:
                    continue

                errors.append(
                    f"WARN: {py_file} contains {description}. "
                    f"Consider using shared infrastructure (base_trainer.py or trainer.py)."
                )

    return errors


def check_shell_scripts() -> list[str]:
    """Check for shell script wrappers around Python scripts."""
    errors = []
    project_root = Path()

    for shell_file in project_root.glob("**/*.ps1"):
        if ".git" in str(shell_file):
            continue
        errors.append(
            f"ERROR: Shell script found: {shell_file}. "
            f"Prefer Python scripts for cross-platform compatibility."
        )

    for shell_file in project_root.glob("**/*.sh"):
        if ".git" in str(shell_file):
            continue
        errors.append(
            f"WARN: Shell script found: {shell_file}. " f"Consider using Python for complex logic."
        )

    return errors


def check_benchmark_proliferation() -> list[str]:
    """Check for multiple benchmark runners."""
    errors = []
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return errors

    benchmark_files = []
    for py_file in experiments_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        if "benchmark" in py_file.name.lower() or "comprehensive" in py_file.name.lower():
            benchmark_files.append(py_file)

    if len(benchmark_files) > 1:
        errors.append(
            f"ERROR: Multiple benchmark files found: {[f.name for f in benchmark_files]}. "
            f"Should have only one: run_benchmark.py"
        )

    return errors


def check_orphaned_standalone_scripts() -> list[str]:
    """Check for standalone training scripts that should use train.py."""
    errors = []
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return errors

    allowed_standalone = {
        "train.py",
        "run_benchmark.py",
        "precompute_audio.py",
        "arg_parser.py",
        "__init__.py",
        "train_audio_precomputed.py",  # Special case with precomputed data
    }

    for py_file in experiments_dir.glob("*.py"):
        if py_file.name in allowed_standalone:
            continue

        content = py_file.read_text(encoding="utf-8", errors="ignore")

        # Check if it's a standalone training script
        if "if __name__" in content and ("train" in py_file.name or "sweep" in py_file.name):
            errors.append(
                f"WARN: Standalone training script found: {py_file.name}. "
                f"Consider using train.py with domain adapters."
            )

    return errors


def check_documentation_duplicates() -> list[str]:
    """Check for duplicate documentation files."""
    errors = []
    docs_dir = Path("docs")
    if not docs_dir.exists():
        return errors

    # Check for multiple versions of the same doc
    base_names = {}
    for doc_file in docs_dir.glob("*.md"):
        # Strip version suffixes and common variations
        base = doc_file.stem.lower()
        for suffix in ["_v2", "_v3", "_old", "_new", "_comprehensive"]:
            base = base.replace(suffix, "")

        if base in base_names:
            errors.append(
                f"WARN: Possible duplicate docs: {base_names[base]} and {doc_file.name}. "
                f"Consider consolidating."
            )
        else:
            base_names[base] = doc_file.name

    return errors


def main() -> int:
    """Run all architecture checks."""
    print("=" * 60)
    print("Polar CV-KAN Architecture Validation")
    print("=" * 60)

    all_errors = []

    print("\n[1/5] Checking for duplicate training loops...")
    errors = check_duplicate_training_loops()
    all_errors.extend(errors)
    print(f"  Found {len(errors)} issues")

    print("\n[2/5] Checking for shell script wrappers...")
    errors = check_shell_scripts()
    all_errors.extend(errors)
    print(f"  Found {len(errors)} issues")

    print("\n[3/5] Checking for benchmark proliferation...")
    errors = check_benchmark_proliferation()
    all_errors.extend(errors)
    print(f"  Found {len(errors)} issues")

    print("\n[4/5] Checking for orphaned standalone scripts...")
    errors = check_orphaned_standalone_scripts()
    all_errors.extend(errors)
    print(f"  Found {len(errors)} issues")

    print("\n[5/5] Checking for documentation duplicates...")
    errors = check_documentation_duplicates()
    all_errors.extend(errors)
    print(f"  Found {len(errors)} issues")

    print("\n" + "=" * 60)

    if all_errors:
        print(f"\nTotal issues found: {len(all_errors)}\n")
        for error in all_errors:
            prefix = "🔴" if error.startswith("ERROR") else "🟡"
            print(f"  {prefix} {error}")

        # Count errors vs warnings
        error_count = sum(1 for e in all_errors if e.startswith("ERROR"))
        warn_count = sum(1 for e in all_errors if e.startswith("WARN"))

        print(f"\nSummary: {error_count} errors, {warn_count} warnings")
        return 1 if error_count > 0 else 0
    print("\n✅ All architecture checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
