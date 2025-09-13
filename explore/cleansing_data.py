import os
import shutil
import pandas as pd

# Folder asal dan folder tujuan
src_folder = "./data/stock_data/stocks"
dst_folder = "./data/stock_data/stocks_removed"
lookup_file = "./tables/stock_lookup.csv"
analyst_file = "./data/analyst_ratings_processed.csv"

# Buat folder tujuan kalau belum ada
os.makedirs(dst_folder, exist_ok=True)

# Daftar ticker yang valid dari dataframe
tables_df = pd.read_csv(lookup_file)
valid_tickers = set(tables_df["stock"].unique())

# Loop semua file di folder asal
for filename in os.listdir(src_folder):
    if filename.endswith(".csv"):
        ticker = filename.replace(".csv", "")
        
        # Kalau ticker tidak ada di daftar valid → pindahkan
        if ticker not in valid_tickers:
            src_path = os.path.join(src_folder, filename)
            dst_path = os.path.join(dst_folder, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved: {filename} → {dst_folder}")

print("Cleanup done ✅")

# Ambil daftar ticker valid dari nama file di ./data/stocks
valid_tickers = {f.replace(".csv", "") for f in os.listdir(src_folder) if f.endswith(".csv")}

# Load datasets
analyst_df = pd.read_csv(analyst_file)
lookup_df = pd.read_csv(lookup_file)

# Filter hanya ticker yang ada di folder stocks
analyst_filtered = analyst_df[analyst_df["stock"].isin(valid_tickers)].copy()
lookup_filtered = lookup_df[lookup_df["stock"].isin(valid_tickers)].copy()

# Simpan hasil (opsional)
analyst_filtered.to_csv("./data/analyst_ratings_filtered.csv", index=False)
lookup_filtered.to_csv("./tables/stock_lookup_filtered.csv", index=False)

print("Rows sebelum & sesudah:")
print("Analyst:", len(analyst_df), "→", len(analyst_filtered))
print("Lookup :", len(lookup_df), "→", len(lookup_filtered))