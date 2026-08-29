import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.feature_pipeline import run_feature_pipeline

def backfill_historical_data(days: int = 365):
    """
    Run feature pipeline over extended historical window (365 days / 1 full year) to populate training dataset.
    """
    print(f"==================================================")
    print(f" Starting 1-Year Historical Data Backfill for {days} Days")
    print(f" Target Cities: Karachi, Chicago, Sydney, Austria (Vienna)")
    print(f"==================================================")
    
    df = run_feature_pipeline(past_days=days)
    print(f"\n[COMPLETE] 1-Year Backfill finished! Total rows generated: {len(df)}")
    return df

if __name__ == "__main__":
    backfill_historical_data(days=365)
