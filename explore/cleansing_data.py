import os
import shutil
import pandas as pd

# Paths
SRC_FOLDER = "./data/stock_data/stocks"
DST_FOLDER = "./data/stock_data/stocks_removed"
LOOKUP_FILE = "./tables/stock_lookup.csv"
ANALYST_FILE = "./data/analyst_ratings_processed.csv"

def ensure_folder(path: str):
    """Create folder if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def move_invalid_files(src_folder: str, dst_folder: str, valid_tickers: set):
    """Move CSV files not in valid tickers to dst_folder."""
    ensure_folder(dst_folder)
    for filename in os.listdir(src_folder):
        if filename.endswith(".csv"):
            ticker = filename.replace(".csv", "")
            if ticker not in valid_tickers:
                src_path = os.path.join(src_folder, filename)
                dst_path = os.path.join(dst_folder, filename)
                shutil.move(src_path, dst_path)
                print(f"Moved: {filename} → {dst_folder}")
    print("File cleanup done ✅")

def filter_datasets(analyst_file: str, lookup_file: str, valid_tickers: set):
    """Filter analyst and lookup datasets to only keep valid tickers."""
    analyst_df = pd.read_csv(analyst_file)
    lookup_df = pd.read_csv(lookup_file)

    analyst_filtered = analyst_df[analyst_df["stock"].isin(valid_tickers)].copy()
    lookup_filtered = lookup_df[lookup_df["stock"].isin(valid_tickers)].copy()

    analyst_filtered.to_csv("./data/analyst_ratings_filtered.csv", index=False)
    lookup_filtered.to_csv("./tables/stock_lookup_filtered.csv", index=False)

    print("Rows sebelum & sesudah:")
    print("Analyst:", len(analyst_df), "→", len(analyst_filtered))
    print("Lookup :", len(lookup_df), "→", len(lookup_filtered))

def main():
    # Load valid tickers from lookup table
    tables_df = pd.read_csv(LOOKUP_FILE)
    valid_tickers = set(tables_df["stock"].dropna().astype(str).unique())

    # Step 1: Move invalid stock files
    move_invalid_files(SRC_FOLDER, DST_FOLDER, valid_tickers)

    # Step 2: Refresh valid tickers from remaining stock files
    valid_tickers = {f.replace(".csv", "") for f in os.listdir(SRC_FOLDER) if f.endswith(".csv")}

    # Step 3: Filter datasets
    filter_datasets(ANALYST_FILE, LOOKUP_FILE, valid_tickers)

if __name__ == "__main__":
    main()