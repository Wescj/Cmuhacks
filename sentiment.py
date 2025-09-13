# classify_sentiment.py
# pip install -U google-genai pandas
# export GEMINI_API_KEY="YOUR_KEY"

from google import genai
from google.genai import types
from google.genai.errors import ClientError
import pandas as pd
import os
import sys
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

INPUT_CSV  = "data/analyst_ratings_filtered.csv"                 # change if needed
# INPUT_CSV  = "tables/test_sample.csv" 
OUTPUT_CSV = "tables/test_sample_with_sentiment.csv"  # output path
MODEL = "gemini-2.5-flash"                          # fast & inexpensive

#Vader calls
nltk.download("vader_lexicon", quiet=True) 
_sia = SentimentIntensityAnalyzer()

PROMPT_SYS = (
    "You are a precise financial headline sentiment rater. "
    "For each given title, return STRICT JSON with fields: "
    "{\"label\": \"positive|neutral|negative\"}. "
    "Base the label on the likely short-term price reaction for the mentioned stock."
)

def classify_title(client, title: str) -> str:
    cfg = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
    )
    contents = [
        {"text": PROMPT_SYS},
        {"text": f"Title:\n- {title}"}
    ]

    try:
        resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
        txt = resp.candidates[0].content.parts[0].text
        print(f"Title: \n{title} \n Output: {txt}\n")
        parsed = json.loads(txt) if txt.strip().startswith(("{", "[")) else {"label": txt}
        if isinstance(parsed, dict):
            label = str(parsed.get("label", "")).lower().strip()
        elif isinstance(parsed, list) and parsed:
            label = str(parsed[0].get("label", "")).lower().strip()
        else:
            return "unknown"
    except (ClientError, json.JSONDecodeError, Exception):
        return "unknown"

    if label.startswith("pos"):
        return "positive"
    if label.startswith("neg"):
        return "negative"
    if label.startswith("neu"):
        return "neutral"
    return "unknown"

def classify_title_vader(title: str) -> str:
    """
    Return 'positive' | 'neutral' | 'negative' using VADER compound score.
    Thresholds follow the VADER paper/repo convention:
      compound >=  0.05 -> positive
      compound <= -0.05 -> negative
      otherwise         -> neutral
    """
    print( f"Title: \n{title}\n" )
    if title is None:
        return "neutral"
    scores = _sia.polarity_scores(str(title))
    c = scores.get("compound", 0.0)
    if c >= 0.05:
        return "positive"
    if c <= -0.05:
        return "negative"
    return "neutral"

def main():
    # Check API key
    print("Classifying sentiment using Gemini...")
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)
    print("API key found.")
    df = pd.read_csv(INPUT_CSV)
    if "title" not in df.columns:
        print(f"ERROR: '{INPUT_CSV}' must have a 'title' column.", file=sys.stderr)
        sys.exit(1)
    print("Table found.")
    client = genai.Client()

    sentiments = []
    for t in df["title"].astype(str).fillna(""):
        # sentiments.append(classify_title(client, t))
        sentiments.append(classify_title_vader(t))

    #Testing only 2 titles for now
    # for t in df["title"].astype(str).fillna("").head(2):
    #     sentiments.append(classify_title(client, t))

    # create/update the sentiment column
    df["sentiment"] = sentiments

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")
    print(df[["title", "sentiment"]].head())

if __name__ == "__main__":
    main()
