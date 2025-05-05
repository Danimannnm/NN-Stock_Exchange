from pathlib import Path
from train_models import train_and_save
import models_lstm, models_cnn, models_transformer

def main():
    split_dir = Path("data/splits")
    # find every training split file
    for train_path in split_dir.glob("*_train.csv"):   # e.g. AAPL_day_..._train.csv :contentReference[oaicite:2]{index=2}
        # derive val and test paths by name pattern
        base = train_path.stem.replace("_train","")   # e.g. "AAPL_day_..._numeric"
        val_path  = split_dir / f"{base}_val.csv"
        test_path = split_dir / f"{base}_test.csv"

        # ensure files exist
        if not val_path.exists() or not test_path.exists():
            print(f"Skipping {base}: missing val or test split")
            continue

        # run training for each architecture
        print(f"Training on {base}...")
        train_and_save(models_lstm.build_lstm,        str(train_path), str(val_path), f"{base}_lstm")
        train_and_save(models_cnn.build_cnn1d,        str(train_path), str(val_path), f"{base}_cnn1d")
        train_and_save(models_transformer.build_transformer, str(train_path), str(val_path), f"{base}_transformer")

if __name__ == "__main__":
     main()