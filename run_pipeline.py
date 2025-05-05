# run_pipeline.py
import yaml
from src.ingestion.historical_data import fetch_and_save_bars

def load_settings(path="configs/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_settings()
    for sym in config["symbols"]:
        fetch_and_save_bars(
            symbol   = sym,
            interval = config["interval"],
            start    = config["start_date"],
            end      = config["end_date"]
        )
