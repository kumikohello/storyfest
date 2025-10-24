# Authors: Yolanda Pan (xpan02@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: October 24, 2025
# Description: The script helps to generate a list of details/peripherals for different events from event annotations, matching the number of gists per event.

import os, sys, argparse
import openai
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------- env & API key ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

api_key = "OPENAI_API_KEY"

if not api_key:
    sys.exit("[ERROR] OPENAI_API_KEY not found. Put it in .env or export it before running.")
openai.api_key = api_key

# ---- dataset & paths ----
_ap = argparse.ArgumentParser(add_help=False)

REPO = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
DAT_PATH = REPO / "experiment"
if not DAT_PATH.exists():
    sys.exit(f"[ERROR] Not found: {DAT_PATH}")
SAVE_PATH = REPO / "data" / "gist_detail"
if not SAVE_PATH.exists():
    sys.exit(f"[ERROR] Not found: {SAVE_PATH}")
NUM_PATH = SAVE_PATH / "central_details"
if not NUM_PATH.exists():
    sys.exit(f"[ERROR] Not found: {NUM_PATH}")
current_save_path = SAVE_PATH / "peripheral_detail"
os.makedirs(current_save_path, exist_ok=True)

# ------------------ Define functions ------------------ #
PROMPT = '''
    **Task:**
    You are assisting a memory researcher in analyzing an audio story transcript to extract its **peripheral details**.
    
    ---
    **Definition of Peripheral Details:**
    Peripheral details are descriptive elements that enrich the narrative context but are not essential to its causal structure. They provide texture, atmosphere, or background information (e.g., setting descriptions or incidental features), yet their absence would not alter the core storyline or change character motivations.
    ---

    **Central Storyline as Reference:**
    \"\"\" {summary}\"\"\"

    ---

    **Steps:**
    1. Identify distinct details that are **descriptive but not causally essential**.
    2. Exclude any plot-driving events, states, or turning points (those belong in central).
    3. Express each idea in a brief (≤10 words) form that captures its plot-relevant role.
    4. Avoid redundancy or interpretation (no camera notes, no analysis).
    5. Extract **exactly {num_details} details** — no more, no less.

    ---

    **Deliverable:**
    Provide a table with exactly {num_details} peripheral details, formatted like this:

    | Peripheral ID | Detail |
    |---------------|--------|
    | P1            | ...    |
    | P2            | ...    |
    | ...           | ...    |
'''.strip()

def generate_peripheral_details(summary, num_details):
    prompt = PROMPT.format(summary=summary, num_details=num_details)
    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content

def parse_peripheral_detail_table(gpt_output: str, story: Optional[str], event_number=None):
    lines = gpt_output.strip().splitlines()
    details = []
    for line in lines:
        if line.strip().startswith("| P") and "|" in line:
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 2:
                peripheral_id, idea_unit = parts
                details.append({
                    "story": story,
                    "event_number": event_number,
                    "peripheral_id": peripheral_id,
                    "peripheral": idea_unit
                })
    return details

def flatten_peripheral_data(raw_data):
    flat_list = []

    for sublist in raw_data:
        for row in sublist:
            if row["peripheral_id"] != "Peripheral ID":  # Skip the header row
                flat_list.append({
                    "story": row["story"],
                    "event_number": row["event_number"],
                    "peripheral_id": row["peripheral_id"],
                    "peripheral": row["peripheral"]
                })

    return pd.DataFrame(flat_list)

def load_annotations(path: Path, story_col_candidates=("story", "story_name", "Story"),
                     eventnum_col_candidates=("event_number", "event", "EventNumber"),
                     subject_col_candidates=("subject", "Subject", "SID")) -> pd.DataFrame:
    """
    Load Storyfest annotations from:
      - a CSV with columns per event (preferred), OR
      - an Excel with 6 sheets (one sheet per story).
    Must contain at minimum: story, event_number (or equivalents).
    If subject column exists, it will be preserved.
    """
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {path}")

    def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        # Find columns by candidates
        def pick(colnames):
            for c in colnames:
                if c in df.columns:
                    return c
            return None

        story_c = pick(story_col_candidates)
        event_c = pick(eventnum_col_candidates)
        subj_c  = pick(subject_col_candidates)

        missing = []
        if story_c is None: missing.append("story")
        if event_c is None: missing.append("event_number")
        if missing:
            raise ValueError(f"Missing required columns in annotations: {missing}.\nColumns found: {list(df.columns)}")

        out = pd.DataFrame({
            "story": df[story_c],
            "event_number": df[event_c]
        })
        if subj_c is not None:
            out["subject"] = df[subj_c]
        else:
            out["subject"] = pd.NA
        return out

    if path.suffix.lower() in [".xlsx", ".xls"]:
        # Concatenate all sheets; expect each sheet corresponds to a story
        xls = pd.ExcelFile(path)
        dfs = []
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            # If no story column, inject from sheet name
            if not any(c in df.columns for c in story_col_candidates):
                df = df.copy()
                df["story"] = sheet
            dfs.append(_standardize_cols(df))
        annotations = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path)
        annotations = _standardize_cols(df)

    # Event numbers should be int where possible
    with pd.option_context('mode.chained_assignment', None):
        annotations["event_number"] = pd.to_numeric(annotations["event_number"], errors="coerce").astype("Int64")
        annotations["story"] = annotations["story"].astype(str)
        if "subject" in annotations.columns:
            annotations["subject"] = annotations["subject"].astype("string")

    return annotations

def load_summaries(summary_path: Path, story_col_candidates=("story", "Story"),
                   summary_col_candidates=("summary", "Transcript_Summary")) -> pd.DataFrame:
    """
    Load per-story summaries from CSV/JSON.
      CSV: columns [story, summary] (or candidate names)
      JSON: list of { "story": "...", "summary": "..." }
    """
    if not summary_path.exists():
        sys.exit(f"[ERROR] Summary file not found: {summary_path}")

    if summary_path.suffix.lower() in [".json"]:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        df = pd.DataFrame(data)
    else:
        df = pd.read_excel(summary_path)

    def pick(colnames):
        for c in colnames:
            if c in df.columns:
                return c
        return None

    story_c = pick(story_col_candidates)
    summ_c  = pick(summary_col_candidates)
    if story_c is None or summ_c is None:
        raise ValueError(f"Summary file must have story & summary columns.\nColumns found: {list(df.columns)}")

    out = df[[story_c, summ_c]].rename(columns={story_c: "story", summ_c: "summary"}).dropna()
    out["story"] = out["story"].astype(str)
    out["summary"] = out["summary"].astype(str)
    return out

def match_summary(summ_df: pd.DataFrame, story: str) -> Optional[str]:
    # Exact match first
    exact = summ_df.loc[summ_df["story"] == story, "summary"]
    if not exact.empty:
        return exact.iloc[0]
    # Try case-insensitive / trimmed match
    cand = summ_df.loc[summ_df["story"].str.strip().str.lower() == str(story).strip().lower(), "summary"]
    if not cand.empty:
        return cand.iloc[0]
    return None

# ------------------- Main ------------------ #
def main():
    central_files = list(NUM_PATH.glob("*.csv"))
    if not central_files:
        raise FileNotFoundError("No *.csv file found in NUM_PATH")
    central_file = central_files[0]
    
    central_table = pd.read_csv(central_file)

    annotations_path = DAT_PATH / "eventsegmentation_coarse.xlsx"
    summaries_path = DAT_PATH / "Transcript_Summary.xlsx"
    print(f"[INFO] Using annotations: {annotations_path.name}")
    print(f"[INFO] Using summaries:   {summaries_path.name}")

    annotations = load_annotations(annotations_path)
    summaries = load_summaries(summaries_path)
    print(f"[INFO] Loaded {len(annotations)} annotation rows")
    print(f"[INFO] Loaded {len(summaries)} story summaries")

    event_central_counts = defaultdict(int)
    for i, row in central_table.iterrows():
        if pd.isna(row["event_number"]):
            continue
        try:
            event_number = int(row["event_number"])
        except Exception:
            print(f"[WARN] Skipping non-integer event_number at row {i}: {row['event_number']}")
            continue
        event_central_counts[event_number] += 1
    
    peripheral_table_all = []
    for _, row in annotations.iterrows():
        story = str(row["story"])
        event_number = int(row["event_number"]) if pd.notna(row["event_number"]) else None
        summary = match_summary(summaries, story)
        if summary is None:
            print(f"[WARN] No summary found for story='{story}'")
            continue

        num_details = event_central_counts.get(event_number, 6)
        peripheral_table = parse_peripheral_detail_table(generate_peripheral_details(summary, num_details), story, event_number)
        peripheral_table_all.append(peripheral_table)
    
    peripheral_df = flatten_peripheral_data(peripheral_table_all)
    out_path = current_save_path / "storyfest_balanced_peripheral_detail_table.csv"
    peripheral_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {len(peripheral_df)} rows → {out_path}")

if __name__ == "__main__":
    main()