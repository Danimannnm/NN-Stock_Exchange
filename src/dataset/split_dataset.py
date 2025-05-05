# src/dataset/split_dataset.py

import os
import pandas as pd
from typing import Tuple

def split_time_series(
    in_path: str,
    out_dir: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15
) -> Tuple[str, str, str]:
    """
    1. Load a numeric CSV with datetime index.
    2. Split into train/val/test by chronological fractions.
    3. Save three CSVs to out_dir and return their paths.
    """
    # Load data with datetime index
    df = pd.read_csv(
        in_path,
        parse_dates=['datetime'],    # ensure datetime index :contentReference[oaicite:5]{index=5}
        index_col='datetime',
        infer_datetime_format=True
    )

    n = len(df)
    train_end = int(n * train_frac)
    val_end   = train_end + int(n * val_frac)

    train_df = df.iloc[:train_end]    # first 70% :contentReference[oaicite:6]{index=6}
    val_df   = df.iloc[train_end:val_end]  # next 15% :contentReference[oaicite:7]{index=7}
    test_df  = df.iloc[val_end:]      # final ~15% :contentReference[oaicite:8]{index=8}

    # Prepare output paths
    symbol = os.path.basename(in_path).replace('.csv','')
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, f"{symbol}_train.csv")
    val_path   = os.path.join(out_dir, f"{symbol}_val.csv")
    test_path  = os.path.join(out_dir, f"{symbol}_test.csv")

    # Save splits
    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    print(f"Split {symbol}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
    return train_path, val_path, test_path
