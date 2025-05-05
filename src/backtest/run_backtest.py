# src/backtest/run_backtest.py

from pathlib import Path
from backtest_models import backtest_model
import pickle


def main():
    # Directory containing train/val/test split CSVs
    split_dir = Path("data/splits")
    # Directory containing saved model files (.keras)
    model_dir = Path("models")
    # Directory containing saved scalers (pickled)
    scaler_dir = Path("models/scalers")
    # Where to write backtest results
    results_dir = Path("data/backtest")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Loop over every training split file
    for train_path in split_dir.glob("*_train.csv"):  # e.g. AAPL_train.csv
        # Derive base name (e.g. "AAPL")
        base = train_path.stem.replace("_train", "")
        # Construct validation and test paths
        val_path = split_dir / f"{base}_val.csv"
        test_path = split_dir / f"{base}_test.csv"

        # Check that the split files exist
        if not val_path.exists() or not test_path.exists():
            print(f"Missing val or test split for {base}, skipping.")
            continue

        # Load the scaler for this symbol
        scaler_path = scaler_dir / f"{base}_scaler.pkl"
        if not scaler_path.exists():
            print(f"Scaler not found for {base}, skipping.")
            continue
        scaler = pickle.load(open(scaler_path, "rb"))

        # Backtest each model for this symbol
        # Model files should be named like AAPL_lstm.keras, AAPL_cnn1d.keras, etc.
        for model_file in model_dir.glob(f"{base}_*.keras"):  # e.g. AAPL_lstm.keras
            print(f"Backtesting model {model_file.name} on {base}")
            mse, mae, da = backtest_model(
                model_path=str(model_file),
                scaler=scaler,
                test_csv=str(test_path),
                window_size=10,
                results_dir=str(results_dir)
            )


if __name__ == "__main__":
    main()
