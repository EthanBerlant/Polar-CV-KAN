"""
Visualize sentiment analysis behavior.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_sst2 import TextClassifier
from src.data.text import SimpleTokenizer, load_sst2


def analyze_text(model, text, vocab, device="cpu"):
    model.eval()
    tokenizer = SimpleTokenizer()
    tokens = tokenizer.tokenize(text)
    indices = [vocab[t] for t in tokens]

    x = torch.tensor([indices], device=device)
    mask = torch.ones_like(x)

    with torch.no_grad():
        outputs = model(x, mask=mask, return_intermediates=True)

    # Analyze final layer Z
    Z = outputs["intermediates"][-1]  # (1, n_tokens, d_complex)

    # Magnitudes (importance?)
    mags = torch.abs(Z).mean(dim=-1).squeeze().cpu().numpy()

    # Phases
    phases = torch.angle(Z).squeeze().cpu().numpy()  # (n_tokens, d)

    # Coherence across dimensions?
    # Or correlation between tokens?

    print(f"Text: {text}")
    print(f"Prediction: {outputs['logits'].argmax().item()}")
    print("Token Magnitudes:")
    for t, m in zip(tokens, mags, strict=False):
        print(f"  {t}: {m:.4f}")

    return tokens, mags, phases


def main():
    device = "cpu"

    # Load model
    print("Loading model...")
    _, _, vocab = load_sst2(max_len=64)
    model = TextClassifier(
        vocab_size=len(vocab),
        embed_dim=64,
        cvkan_args={
            "d_complex": 64,
            "n_layers": 2,
            "n_classes": 2,
            "kan_hidden": 32,
            "head_approach": "emergent",
            "pooling": "mean",
            "input_type": "real",
        },
    )
    model.load_state_dict(torch.load("outputs/sst2/best.pt", map_location=device))
    model.to(device)
    model.eval()

    examples = [
        "this movie is absolutely fantastic and wonderful",
        "terrible acting and a boring plot makes this a disaster",
        "it was okay but not great",
        "a visual masterpiece with stunning effects",
    ]

    plt.figure(figsize=(10, 6))

    for i, text in enumerate(examples):
        tokens, mags, phases = analyze_text(model, text, vocab, device)

        plt.subplot(2, 2, i + 1)
        plt.bar(tokens, mags)
        plt.xticks(rotation=45)
        plt.title(f"Token Magnitudes\n'{text}'")

    plt.tight_layout()
    plt.savefig("visualizations/sentiment_analysis.png")
    print("Saved visualizations/sentiment_analysis.png")


if __name__ == "__main__":
    main()
