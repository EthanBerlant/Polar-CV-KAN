#!/usr/bin/env python
"""Master runner for all research experiments.

Usage:
    python run_all.py                    # Run all phases (no checkpoint needed)
    python run_all.py --phase 1          # Run only phase 1
    python run_all.py --phase 1 2 3      # Run phases 1, 2, 3
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RESEARCH_DIR = Path(__file__).parent
sys.path.insert(0, str(RESEARCH_DIR))
sys.path.insert(0, str(RESEARCH_DIR.parent))  # Project root for research.phaseX imports

from config import CONFIG

from research.phase6_robustness.run_experiment import run_experiment as run_phase6_exp
from research.phase7_real_world.run_mnist import run_mnist_experiment as run_phase7_exp
from research.phase8_nlp.run_imdb import run_imdb_experiment as run_phase8_exp


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60 + "\n")

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Script not found: {cmd[1]}")
        return False


def run_phase1():
    """Run Phase 1: Polarization Dynamics (no checkpoint needed)."""
    print("\n" + "#" * 60)
    print("# PHASE 1: POLARIZATION DYNAMICS")
    print("#" * 60)

    phase_dir = RESEARCH_DIR / "phase1_polarization"
    results = []

    success = run_command(
        [sys.executable, str(phase_dir / "single_block_analysis.py")],
        "Single block theoretical analysis",
    )
    results.append(("single_block_analysis", success))

    success = run_command(
        [
            sys.executable,
            str(phase_dir / "synthetic_polarization.py"),
            "--epochs",
            "30",
            "--n_layers",
            "4",
        ],
        "Synthetic polarization experiments",
    )
    results.append(("synthetic_polarization", success))

    return results


def run_phase2():
    """Run Phase 2: Phase Semantics (uses synthetic model)."""
    print("\n" + "#" * 60)
    print("# PHASE 2: PHASE SEMANTICS")
    print("#" * 60)
    print("Phase 2 requires a trained checkpoint.")
    print("Run: python research/phase2_semantics/phase_clustering.py --checkpoint <path>")
    return [("phase2", "skipped")]


def run_phase3():
    """Run Phase 3: Complex vs Real ablations."""
    print("\n" + "#" * 60)
    print("# PHASE 3: COMPLEX NECESSITY ABLATIONS")
    print("#" * 60)

    phase_dir = RESEARCH_DIR / "phase3_complex_necessity"

    success = run_command(
        [sys.executable, str(phase_dir / "run_ablations.py"), "--epochs", "10"],
        "Complex vs Real ablations",
    )

    return [("ablations", success)]


def run_phase4():
    """Run Phase 4: KAN Framing (literature only)."""
    print("\n" + "#" * 60)
    print("# PHASE 4: KAN FRAMING ANALYSIS")
    print("#" * 60)

    notes_path = RESEARCH_DIR / "phase4_kan_framing" / "literature_notes.md"

    if notes_path.exists():
        print(f"Literature notes available at: {notes_path}")
        print("This phase is documentation only - no code to run.")
        return [("literature_notes", True)]
    print(f"ERROR: Literature notes not found at {notes_path}")
    return [("literature_notes", False)]


def run_phase5():
    """Run Phase 5: Gradient and Stability Analysis."""
    print("\n" + "#" * 60)
    print("# PHASE 5: GRADIENT AND STABILITY ANALYSIS")
    print("#" * 60)

    phase_dir = RESEARCH_DIR / "phase5_stability"
    results = []

    success = run_command(
        [sys.executable, str(phase_dir / "gradient_flow.py"), "--epochs", "3"],
        "Gradient flow analysis",
    )
    results.append(("gradient_flow", success))

    success = run_command(
        [sys.executable, str(phase_dir / "magnitude_dynamics.py"), "--epochs", "5"],
        "Magnitude dynamics analysis",
    )
    results.append(("magnitude_dynamics", success))

    return results


def run_phase6():
    """Run Phase 6: Robustness (Synthetic Rotated Shapes)."""
    print("\n" + "#" * 60)
    print("# PHASE 6: ROBUSTNESS (ROTATED SHAPES)")
    print("#" * 60)

    args = argparse.Namespace(
        epochs=10, lr=1e-3, n_samples=2000, output_dir="research/outputs/phase6"
    )

    try:
        results = run_phase6_exp(args)
        # Propagate full results dict for synthesis
        return results
    except Exception as e:
        print(f"Phase 6 Failed: {e}")
        return {"error": str(e)}


def run_phase7():
    """Run Phase 7: Real World Vision (MNIST)."""
    print("\n" + "#" * 60)
    print("# PHASE 7: REAL WORLD VISION (MNIST)")
    print("#" * 60)

    args = argparse.Namespace(
        epochs=5, lr=1e-3, n_samples=2000, output_dir="research/outputs/phase7"
    )

    try:
        results = run_phase7_exp(args)
        return results
    except Exception as e:
        print(f"Phase 7 Failed: {e}")
        return {"error": str(e)}


def run_phase8():
    """Run Phase 8: NLP Validation (IMDB)."""
    print("\n" + "#" * 60)
    print("# PHASE 8: NLP VALIDATION (IMDB)")
    print("#" * 60)

    args = argparse.Namespace(
        epochs=5, lr=1e-3, n_samples=2000, output_dir="research/outputs/phase8"
    )

    try:
        results = run_phase8_exp(args)
        return results
    except Exception as e:
        print(f"Phase 8 Failed: {e}")
        return {"error": str(e)}


def generate_synthesis(all_results: dict, output_dir: Path):
    """Generate synthesis document from all results."""
    synthesis_path = output_dir / "synthesis.md"

    lines = [
        "# Polar CV-KAN Theoretical Research: Synthesis",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Execution Summary",
        "",
    ]

    for phase, results in all_results.items():
        lines.append(f"### {phase}")
        if isinstance(results, dict):
            if "error" in results:
                lines.append(f"- [FAIL] {results['error']}")
            else:
                # Dictionary of model results
                lines.append("| Model | Accuracy | Gini | Phase |")
                lines.append("| :--- | :--- | :--- | :--- |")
                for name, res in results.items():
                    acc = res.get("final_acc")
                    if acc is None:
                        acc = res.get("final_test_acc")
                    if acc is None:
                        acc = res.get("final_up")
                    if acc is None:
                        acc = 0.0

                    gini = res.get("final_gini", 0.0)
                    phase_r = res.get("final_phase", 0.0)
                    lines.append(f"| {name} | {acc:.3f} | {gini:.3f} | {phase_r:.3f} |")
        else:
            # Legacy list of (name, status)
            for name, status in results:
                if status == True:
                    status_str = "[PASS]"
                elif status == "skipped":
                    status_str = "[SKIP]"
                else:
                    status_str = "[FAIL]"
                lines.append(f"- {status_str} {name}")
        lines.append("")

    lines.extend(
        [
            "## Key Findings",
            "",
            "*Review results in outputs/phase*/ directories*",
            "",
            "### Phase 1: Polarization Dynamics",
            "- See outputs/phase1/delta_polarization_heatmap.png",
            "- See outputs/phase1/synthetic_results.json",
            "",
            "### Phase 3: Complex Necessity",
            "- See outputs/phase3/ablation_results.json",
            "",
            "### Phase 4: KAN Framing",
            "- See phase4_kan_framing/literature_notes.md",
            "",
            "### Phase 5: Stability",
            "- See outputs/phase5/gradient_flow.png",
            "- See outputs/phase5/magnitude_dynamics.png",
            "",
            "### Phase 6: Robustness (Rotated Shapes)",
            "- See outputs/phase6/robustness_results.json",
            "- See outputs/phase6/robustness_accuracy.png",
            "",
            "### Phase 7: Real World Vision (MNIST)",
            "- See outputs/phase7/mnist_results.json",
            "- See outputs/phase7/mnist_robustness.png",
            "",
            "### Phase 8: NLP Validation (IMDB)",
            "- See outputs/phase8/imdb_results.json",
        ]
    )

    with open(synthesis_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSynthesis written to: {synthesis_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CV-KAN research experiments")
    parser.add_argument(
        "--phase",
        type=int,
        nargs="*",
        default=None,
        help="Specific phases to run (1-8). Default: all",
    )
    args = parser.parse_args()

    phases_to_run = (
        args.phase if args.phase else [1, 3, 4, 5, 6, 7, 8]
    )  # Skip 2 by default (needs checkpoint)

    print("=" * 60)
    print("POLAR CV-KAN THEORETICAL RESEARCH")
    print("=" * 60)
    print(f"Phases to run: {phases_to_run}")
    print("=" * 60)

    all_results = {}

    if 1 in phases_to_run:
        all_results["Phase 1: Polarization"] = run_phase1()

    if 2 in phases_to_run:
        all_results["Phase 2: Semantics"] = run_phase2()

    if 3 in phases_to_run:
        all_results["Phase 3: Complex Necessity"] = run_phase3()

    if 4 in phases_to_run:
        all_results["Phase 4: KAN Framing"] = run_phase4()

    if 5 in phases_to_run:
        all_results["Phase 5: Stability"] = run_phase5()

    if 6 in phases_to_run:
        all_results["Phase 6: Robustness"] = run_phase6()

    if 7 in phases_to_run:
        all_results["Phase 7: Real World Vision"] = run_phase7()

    if 8 in phases_to_run:
        all_results["Phase 8: NLP"] = run_phase8()

    generate_synthesis(all_results, CONFIG.output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("RESEARCH COMPLETE")
    print("=" * 60)

    total = 0
    passed = 0
    skipped = 0

    for phase, results in all_results.items():
        print(f"\n{phase}:")
        if isinstance(results, dict):
            if "error" in results:
                print(f"  [FAIL] {results['error']}")
            else:
                for name, res in results.items():
                    acc = res.get("final_acc")
                    if acc is None:
                        acc = res.get("final_test_acc")
                    if acc is None:
                        acc = res.get("final_up")
                    if acc is None:
                        acc = 0.0
                    print(f"  [PASS] {name} (Acc: {acc:.3f})")
        else:
            for name, status in results:
                total += 1
                if status == True:
                    passed += 1
                    print(f"  [PASS] {name}")
                elif status == "skipped":
                    skipped += 1
                    print(f"  [SKIP] {name}")
                else:
                    print(f"  [FAIL] {name}")

    print(f"\nTotal: {passed}/{total} passed, {skipped} skipped")
    print(f"\nResults in: {CONFIG.output_dir}")


if __name__ == "__main__":
    main()
