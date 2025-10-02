# monthly_avg.py
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_price_sentiment(
    ticker: str,
    price_dir: str = "data/stock_price",
    sent_dir: str = "data/stock_sentiment",
    price_col: str = "Close",
    date_col_price: str = "Date",
    date_col_sent: str = "date",
    sentiment_col_guess = ("avg_sentiment", "sentiment", "label"),
    monthly_price_agg: str = "mean",  # or "last" for end-of-month price
):
    """
    Plots monthly average price (left y) vs. monthly average sentiment % (right y) for a ticker.

    - Price CSV:  data/stock_price/<TICKER>.csv
        Expected columns include [Date, Close] by default (customizable via args).
    - Sent CSV:   data/stock_sentiment/<TICKER>.csv
        Must include a datetime column and a sentiment column; sentiment can be numeric in [-1, 1]
        or labels {positive, neutral, negative}.
    """

    price_path = os.path.join(price_dir, f"{ticker}.csv")
    sent_path  = os.path.join(sent_dir,  f"{ticker}.csv")

    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Missing price file: {price_path}")
    if not os.path.exists(sent_path):
        raise FileNotFoundError(f"Missing sentiment file: {sent_path}")

    # --- Load & prepare PRICE ---
    p = pd.read_csv(price_path)
    if date_col_price not in p.columns:
        raise ValueError(f"Price file must contain '{date_col_price}' column")
    if price_col not in p.columns:
        raise ValueError(f"Price file must contain '{price_col}' column")

    # Parse price dates (no tz) and normalize to plain date
    p[date_col_price] = pd.to_datetime(p[date_col_price], errors="coerce")
    p = p.dropna(subset=[date_col_price]).copy()
    p = p[[date_col_price, price_col]].rename(columns={date_col_price: "Date"})
    p["Date"] = p["Date"].dt.date  # plain Python date objects

    # --- Load & prepare SENTIMENT ---
    s = pd.read_csv(sent_path)

    # Choose the sentiment column
    sent_col = next((c for c in sentiment_col_guess if c in s.columns), None)
    if sent_col is None:
        raise ValueError(f"Sentiment file must contain one of {sentiment_col_guess}")

    # Find a date/datetime column for sentiment
    if date_col_sent not in s.columns:
        for alt in ("Date", "datetime", "Datetime", "time", "published", "Timestamp"):
            if alt in s.columns:
                date_col_sent = alt
                break
    if date_col_sent not in s.columns:
        raise ValueError("Could not find a date/datetime column in sentiment file")

    # Parse with utc=True to handle mixed TZ; then convert to US/Eastern and drop time → plain dates
    s[date_col_sent] = pd.to_datetime(s[date_col_sent], errors="coerce", utc=True)
    s = s.dropna(subset=[date_col_sent]).copy()
    s = s[[date_col_sent, sent_col]].rename(columns={date_col_sent: "Date", sent_col: "sent"})

    # Map label sentiment → numeric if needed
    if s["sent"].dtype == object:
        mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        s["sent"] = s["sent"].astype(str).str.lower().map(mapping)

    # Coerce to numeric (unmapped strings -> NaN)
    s["sent"] = pd.to_numeric(s["sent"], errors="coerce")

    # Convert timezone-aware datetimes → US/Eastern date
    s["Date"] = s["Date"].dt.tz_convert("US/Eastern").dt.date

    # Aggregate multiple sentiment rows per day to a single daily value
    s_daily = s.groupby("Date", as_index=False)["sent"].mean()

    # --- Merge on Date (both plain dates) ---
    df = (
        p.merge(s_daily, on="Date", how="inner") #left join if you want to see full price
        .sort_values("Date")
        .set_index("Date")
    )

    # make the index a DatetimeIndex (required for resample)
    df.index = pd.to_datetime(df.index)  # tz-naive is fine here


    # --- Monthly aggregation ---
    agg = {
        price_col: monthly_price_agg,   # "mean" or "last"
        "sent": "mean",
    }
    monthly = df.resample("M").agg(agg).dropna(subset=[price_col])

    # Convert sentiment [-1,1] to percentage
    monthly["sent_pct"] = monthly["sent"].fillna(0.0) * 100.0

    # Focus on recent data only
    monthly = monthly.loc[monthly.index >= "2019-01-01"].copy()

    # === Plot 1: Raw Sentiment % ===
    fig, ax1 = plt.subplots(figsize=(10,5))

    # Stock price line (blue)
    ax1.plot(monthly.index, monthly[price_col], color="blue", label="Avg Monthly Price")
    ax1.set_ylabel("Average Monthly Price", color="blue")
    ax1.set_xlabel("Month")

    # Sentiment line (orange)
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["sent_pct"], color="orange", label="Avg Sentiment (%)")
    ax2.set_ylabel("Avg Sentiment (%)", color="orange")

    plt.title(f"{ticker}: Monthly {monthly_price_agg.capitalize()} Price vs. Sentiment (%)")
    fig.tight_layout()
    plt.show()

    # === Plot 2: Sentiment aligned to first stock price ===
    # p0 = monthly[price_col].iloc[0]      # first stock price
    # s0 = monthly["sent_pct"].iloc[0]     # first sentiment percentage
    # if abs(s0) < 1e-9:                   # avoid divide-by-zero
    #     s0 = 1.0

    # # scale + shift sentiment so it starts at same value as stock price
    # monthly["sent_aligned"] = monthly["sent_pct"] * (p0 / s0)

    # fig, ax = plt.subplots(figsize=(10,5))

    # # Stock price line (blue)
    # ax.plot(monthly.index, monthly[price_col], color="blue", linewidth=2, label="Price")

    # # Sentiment line (green, aligned to price’s first point)
    # ax.plot(monthly.index, monthly["sent_aligned"], color="green", linewidth=2, label="Sentiment (aligned)")

    # ax.set_ylabel("Price / Aligned Sentiment")
    # ax.set_xlabel("Month")
    # plt.title(f"{ticker}: Price vs Sentiment (aligned to start)")
    # plt.legend()
    # fig.tight_layout()
    # plt.show()

    return monthly

if __name__ == "__main__":
    # Example
    monthly_nvda = plot_monthly_price_sentiment("NVDA")
    # print(monthly_nvda.head())
