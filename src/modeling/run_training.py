# src/modeling/run_training.py

from pathlib import Path
from train_models import train_and_save
import models_lstm, models_cnn, models_transformer


def main():
    split_dir = Path("data/splits")

    # Loop over each train split
    for train_path in split_dir.glob("*_train.csv"):  # e.g. AAPL_day_..._train.csv
        base = train_path.stem.replace("_train", "")  # e.g. "AAPL_day_..."
        val_path = split_dir / f"{base}_val.csv"

        # Derive a short symbol (before first underscore)
        symbol = base.split("_")[0]

        # Print which symbol we're training
        print(f"Training on {symbol}...")

        # Run training for each architecture with new naming
        train_and_save(models_lstm.build_lstm,
                       str(train_path), str(val_path),
                       f"{symbol}_lstm")
        train_and_save(models_cnn.build_cnn1d,
                       str(train_path), str(val_path),
                       f"{symbol}_cnn1d")
        train_and_save(models_transformer.build_transformer,
                       str(train_path), str(val_path),
                       f"{symbol}_transformer")

if __name__ == "__main__":
    main()
