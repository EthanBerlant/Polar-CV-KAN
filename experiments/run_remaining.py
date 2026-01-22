"""
Run only the remaining experiments.
Reads from remaining_experiments.txt and executes each command.
"""

import subprocess
import sys


def run_remaining():
    with open("experiments/remaining_experiments.txt") as f:
        lines = f.readlines()

    commands = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

    print(f"Found {len(commands)} remaining experiments to run")
    print("=" * 50)

    for i, cmd in enumerate(commands, 1):
        print(f"\n[{i}/{len(commands)}] Running: {cmd[:80]}...")
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"✓ Completed {i}/{len(commands)}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            # Continue with next experiment
            continue
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("All remaining experiments completed!")


if __name__ == "__main__":
    run_remaining()
