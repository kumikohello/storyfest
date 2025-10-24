# Authors: Yolanda Pan (xpan02@uchicago.edu), Kumiko Ueda (kumiko@uchicago.edu)
# Last Edited: October 24, 2025
# Description: The script helps to generate scores for different participants of different events, for memory of central and peripheral details.

import os, sys, argparse
import openai
import pandas as pd
from pathlib import Path

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
_ap.add_argument("--mem-type", dest="mem_type", choices=["central", "peripheral"],
                default=os.getenv("MEM_TYPE", "central"))
_args, _ = _ap.parse_known_args()

MEM_TYPE = _args.mem_type

REPO = Path(__file__).resolve().parents[3] if "__file__" in globals() else Path.cwd()
DAT_PATH = REPO / "experiment"
if not DAT_PATH.exists():
    sys.exit(f"[ERROR] Not found: {DAT_PATH}")

RECALL_PATH = REPO / "data" / "recall_transcripts" / "transcripts_resegmented"
if not RECALL_PATH.exists():
    sys.exit(f"[ERROR] Not found: {RECALL_PATH}")

if MEM_TYPE == "central":
    DETAIL_PATH = REPO / "data" / "gist_detail" / "central_details"
else:
    DETAIL_PATH = REPO / "data" / "gist_detail" / "peripheral_detail"
if not DETAIL_PATH.exists():
    sys.exit(f"[ERROR] Not found: {DETAIL_PATH}")

SAVE_PATH = REPO / "data" / "gist_detail"
if not SAVE_PATH.exists():
    sys.exit(f"[ERROR] Not found: {SAVE_PATH}")

current_save_path = SAVE_PATH / "memory_fidelity" / f'{MEM_TYPE}_detail_scores'
os.makedirs(current_save_path, exist_ok=True)

# ------------------ Define functions ------------------ #
PROMPT_CEN = """
    You are an expert annotator evaluating whether a participant recalled **central details** from listening to an audio story.

    ---

    ### Participant Recall
    Participant `{participant_id}` recalled the following for **Story `{story}`**, Event `{event_number}`:
    \"\"\"{participant_recall}\"\"\"

    ---

    ### Central Details for Story `{story}`, Event {event_number}
    These are plot-essential facts or events. Your task is to assess **how accurately** each central detail is reflected in the participant's recall.

    Use the following scoring scale:

    - **2** = Present: Clearly conveyed in the participant’s recall.
    - **1** = Partially Present: Partially conveyed or ambiguous.
    - **0** = Absent: Not mentioned or implied.

    ---

    **Central Detail Table:**
    {central_details}

    ---
    ### Instructions:
    Return **only** a Markdown table with these columns: `participants_id`, `story`, `event_number`, `central_id`, `score`

    Format the output like this:

    | participants_id | story | event_number | central_id | score |
    |-----------------|--------|--------------|-------------|-------|
    | {participant_id} | {story} | {event_number} | C1 | ? |
    | {participant_id} | {story} | {event_number} | C2 | ? |
    | ...      | ...             | ...          | ...       | ...   |
    """.strip()

PROMPT_PERI = """
    You are an expert annotator evaluating whether a participant recalled **peripheral details** from listening to an audio story.

    ---

    ### Participant Recall
    This is what Participant `{participant_id}` remembered for **Story `{story}`**, Event `{event_number}`:
    \"\"\"{participant_recall}\"\"\"

    ---


    ### Peripheral Details for Story `{story}`, Event {event_number}
    These are **minor, descriptive features** of the event. They do **not change the plot**, but add context or sensory richness. Examples may include expressions, minor gestures, positioning, or manner of action.

    Use the following scoring scale:

    - **2 = Present**: The detail is clearly described in the participant's recall.
    - **1 = Partially Present**: The detail is vaguely or partially mentioned.
    - **0 = Absent**: The detail is not mentioned or implied at all.

    Detail Table:
    {peripheral_details}

    ### Instructions:
    Return **only** a Markdown table with these columns: `participants_id`, `story`, `event_number`, `peripheral_id`, `score`

    Format the table like this:

    | participants_id | story | event_number | peripheral_id | score |
    |-----------------|--------|--------------|----------------|-------|
    | {participant_id} | {story} | {event_number} | P1 | ? |
    | {participant_id} | {story} | {event_number} | P2 | ? |
    | ...      | ...             | ...          | ...       | ...   |
    """.strip()

def generate_graded_central_scores(participant_id, story, participant_recall, event_number, central_details):
    prompt = PROMPT_CEN.format(participant_id=participant_id, story=story, participant_recall=participant_recall, event_number=event_number, central_details=central_details)

    response = openai.chat.completions.create(
      model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

def generate_graded_peripheral_scores(participant_id, story, participant_recall, event_number, peripheral_details):
    prompt = PROMPT_PERI.format(participant_id=participant_id, story=story, participant_recall=participant_recall, event_number=event_number, peripheral_details=peripheral_details)

    response = openai.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

def parse_central_score_table(gpt_output: str):
    lines = gpt_output.strip().splitlines()
    scores = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Detect start of table
        if line.startswith("| participants_id"):
            in_table = True
            continue  # skip header row

        # Skip the separator row (e.g., |----|----|)
        if in_table and line.startswith("|--"):
            continue

        # Parse table rows
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 5:
                participant_id, story, event_number, central_id, score = parts
                scores.append({
                    "participant_id": participant_id,
                    "story": story,
                    "event_number": event_number,
                    "central_id": central_id,
                    "score": int(score) if score.isdigit() else score
                })

    return scores

def parse_peripheral_score_table(gpt_output: str):
    lines = gpt_output.strip().splitlines()
    scores = []
    in_table = False

    for line in lines:
        line = line.strip()

        # Detect start of table
        if line.startswith("| participants_id"):
            in_table = True
            continue  # skip header row

        # Skip the separator row (e.g., |----|----|)
        if in_table and line.startswith("|--"):
            continue

        # Parse table rows
        if in_table and line.startswith("|"):
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) == 5:
                participant_id, story, event_number, peripheral_id, score = parts
                scores.append({
                    "participant_id": participant_id,
                    "story": story,
                    "event_number": event_number,
                    "peripheral_id": peripheral_id,
                    "score": int(score) if score.isdigit() else score
                })

    return scores

def read_recall_file(file_path):
    df = pd.read_excel(file_path)
    participant_id = os.path.basename(file_path).split('_recall_')[0]
    return df, participant_id

def parse_recall(df):
    transcript_by_event = []
    for _, row in df.iterrows():
        if pd.isna(row.get("event_number")):
            continue
        story = str(row.get("story")) if pd.notna(row.get("story")) else None
        event_number = row.get("event_number")
        transcript = row.get("Subj_Transcript")
        transcript_by_event.append((story, event_number, transcript))
    return transcript_by_event

def parse_table_by_event(table, event_number):
    parsed_table = table[table["event_number"] == event_number]
    return parsed_table


# ------------------- Main ------------------ #
def main():
    if MEM_TYPE == "central":
        detail_df = pd.read_csv(DETAIL_PATH / "Storyfest_central_detail_table.csv")
    else:
        detail_df = pd.read_csv(DETAIL_PATH / "storyfest_balanced_peripheral_detail_table.csv")

    recall_files = list(RECALL_PATH.glob("*.xlsx"))

    id_col = 'central_id' if MEM_TYPE == 'central' else 'peripheral_id'

    # iterate over participants （files）
    all_results = []
    for recall_path in recall_files:
        df, participant_id = read_recall_file(recall_path)
        transcript_by_event = parse_recall(df)
        number_events = len(transcript_by_event)
        print(f"{participant_id}: {number_events} events")

        # iterate over events
        results = []
        for story, event_number, participant_recall in transcript_by_event:
            detail_table = parse_table_by_event(detail_df, event_number)

            # Skip events without recalls and record them as 0
            if pd.isna(participant_recall):
                if not detail_table.empty:
                    for did in detail_table[id_col].astype(str).tolist():
                        results.append({
                            "participant_id": participant_id,
                            "story": story,
                            "event_number": event_number,
                            id_col: did,
                            "score": 0
                        })
                continue
            
            if MEM_TYPE == 'central':
                gpt_output = generate_graded_central_scores(participant_id, story, participant_recall, event_number, detail_table)
                output = parse_central_score_table(gpt_output)
            else:
                gpt_output = generate_graded_peripheral_scores(participant_id, story, participant_recall, event_number, detail_table)
                output = parse_peripheral_score_table(gpt_output)
            for row in output:
                row["participant_id"] = participant_id
                row["story"] = story
                row["event_number"] = event_number
            results.extend(output)

        results_df = pd.DataFrame(results)
        print(f"{participant_id} done")
        all_results.append(results_df)
    
    all_combined = pd.concat(all_results, ignore_index=True)
    all_combined = all_combined.sort_values(by=["participant_id", "story", "event_number"])
    all_combined.to_csv(f"{current_save_path}/graded_{MEM_TYPE}_scores_compiled.csv", index=False)
    print("All participant scores saved to one CSV.")

if __name__ == "__main__":
    main()
