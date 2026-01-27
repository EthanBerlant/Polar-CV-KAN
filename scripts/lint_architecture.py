import ast
import os
import sys
from pathlib import Path


def check_architecture_compliance(src_dir: str = "src") -> None:
    """Checks that core components are not instantiated directly but via Registry.

    Rules:
    1. No direct instantiation of 'CVKANBackbone' outside of factory/backbone itself.
    2. No direct instantiation of 'ImageEmbedding' etc outside of factory.
    3. 'experiments/' should not import 'src.modules' directly if possible (should use config).
    """
    print(f"Linting architecture compliance in {src_dir}...")
    errors = []

    for root, _, files in os.walk(src_dir):
        for file in files:
            if not file.endswith(".py"):
                continue

            path = Path(root) / file
            try:
                with path.open(encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(path))
            except Exception as e:
                print(f"Could not parse {path}: {e}")
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id
                    if (
                        name in ["CVKANBackbone", "ImageEmbedding"]
                        and "factory" not in str(path)
                        and "factories" not in str(path)
                        and "backbone" not in str(path)
                        and "cv_kan" not in str(path)
                        and "embeddings" not in str(path)
                    ):
                        errors.append(
                            f"{path}:{node.lineno} - Direct instantiation of '{name}' forbidden. Use Factories/Registry."
                        )

    if errors:
        print("Architecture Compliance Failed:")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("Architecture Compliance Passed.")


if __name__ == "__main__":
    check_architecture_compliance()
