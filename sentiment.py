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

INPUT_CSV  = "tables/test_sample.csv"                 # change if needed
OUTPUT_CSV = "tables/test_sample_with_sentiment.csv"  # output path
MODEL = "gemini-2.5-flash"                          # fast & inexpensive

PROMPT_SYS = (
    "You are a precise financial headline sentiment rater. "
    "For each given title, return STRICT JSON with fields: "
    "{\"label\": \"positive|neutral|negative\"}. "
    "Base the label on the likely short-term price reaction for the mentioned stock."
)
# def classify_title(client, title: str) -> str:
#     cfg = types.GenerateContentConfig(
#         temperature=0,
#         response_mime_type="application/json",
#         thinking_config=types.ThinkingConfig(thinking_budget=0)  # no chain-of-thought can change but unnecessary for now
#     )
#     contents = [
#         {"text": PROMPT_SYS},
#         {"text": f"Title:\n- {title}"}
#     ]
#     resp = client.models.generate_content(model=MODEL, contents=contents, config=cfg)
    
#     txt = resp.candidates[0].content.parts[0].text
#     print(f"Title: \n{title} \n DEBUG: resp={resp.parsed} \n Output: {txt}\n") 
#     try:
#         label = json.loads(txt)["label"].lower().strip()
#     except Exception:
#         label = txt.lower().strip()
#     # normalize to one of the three
#     if label.startswith("pos"):
#         return "positive"
#     if label.startswith("neg"):
#         return "negative"
#     return "neutral"

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


def main():
    # Check API key
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY before running.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    if "title" not in df.columns:
        print(f"ERROR: '{INPUT_CSV}' must have a 'title' column.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client()

    sentiments = []
    for t in df["title"].astype(str).fillna(""):
        sentiments.append(classify_title(client, t))

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
