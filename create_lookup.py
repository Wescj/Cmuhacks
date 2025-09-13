import pandas as pd
# Found 6192 unique stocks

# 1) Load your unique stock ids from news
news = pd.read_csv("data/analyst_ratings_processed.csv")
unique_stocks = sorted(news["stock"].dropna().unique())

# 2) Load metadata
meta = pd.read_csv("data/stock_data/symbols_valid_meta.csv")

# Standardize column names just in case
meta = meta.rename(columns={
    "Symbol": "stock",
    "Security Name": "security_name"
})

# 3) Build lookup table
lookup = pd.DataFrame({"stock": unique_stocks})
lookup = lookup.merge(meta, on="stock", how="inner")

# Add an index column
lookup.reset_index(inplace=True)
lookup.rename(columns={"index": "id"}, inplace=True)

# 4) Save to file
lookup.to_csv("data/stock_lookup.csv", index=False)

print(lookup.head())

