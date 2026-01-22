import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.baselines.lstm_text_baseline import TextLSTM
from experiments.baselines.resnet_baseline import ResNetSmall
from experiments.baselines.transformer_audio_baseline import AudioTransformer
from experiments.baselines.transformer_ts_baseline import TransformerForecaster


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    print("Checking Baseline Parameters...")

    # 1. Image: ResNetSmall (CIFAR-10)
    # n_classes=10
    resnet = ResNetSmall(num_classes=10, base_channels=32)
    print(f"Image (ResNetSmall): {count(resnet):,}")

    # 2. NLP: TextLSTM (SST-2)
    # vocab ~16K-20K usually. SST-2 vocab is likely ~15K.
    # Let's assume 15000 for estimate, or we need to load vocab.
    # Args from baseline file: embed=128, hidden=128, layers=2
    lstm = TextLSTM(vocab_size=15000, embed_dim=128, hidden_dim=128, num_layers=2, num_classes=2)
    print(f"NLP (LSTM, vocab=15k): {count(lstm):,}")

    # Exclude embedding from comparison?
    # Usually we care about model capacity. But embedding is huge.
    # Let's print non-embedding params too.
    lstm_non_emb = count(lstm) - count(lstm.embedding)
    print(f"NLP (LSTM, no emb):  {lstm_non_emb:,}")

    # 3. Audio: Transformer (Speech Commands)
    # n_mels=64, n_classes=35, d_model=128, layers=4
    audio = AudioTransformer(n_mels=64, num_classes=35, d_model=128, num_layers=4)
    print(f"Audio (Transformer): {count(audio):,}")

    # 4. TimeSeries: Transformer (ETTh1)
    # input_dim=7, pred_len=96, d_model=128, layers=4
    ts = TransformerForecaster(input_dim=7, output_dim=7, pred_len=96, d_model=128, num_layers=4)
    print(f"TimeSeries (Transformer): {count(ts):,}")


if __name__ == "__main__":
    main()
