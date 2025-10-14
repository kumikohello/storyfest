import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# # --- Setup ---
# excel_path = '/Users/UChicago/CASNL/storyfest/experiment/Storyfest_Event_Segmentation.xlsx'
# save_dir = '/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/16_sentiment_by_event_transformer/vader'
# os.makedirs(save_dir, exist_ok=True)


# --- Setup ---
excel_path = '/Users/UChicago/CASNL/storyfest/storyfest/experiment/Storyfest_Event_Segmentation.xlsx'
save_dir = '/Users/UChicago/CASNL/storyfest/storyfest/data/pupil/3_processed/16_sentiment_by_event_transformer/transformer/story_wise'  # Updated directory
os.makedirs(save_dir, exist_ok=True)

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None, device=-1)

# Label mapping for the transformer output
LABEL_MAPPING = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

# --- Read all sheets ---
xls = pd.ExcelFile(excel_path)

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)

    event_col = next((col for col in df.columns if 'event' in col.lower() and 'number' in col.lower()), None)
    text_col = next((col for col in df.columns if 'transcript' in col.lower()), None)

    if not event_col or not text_col:
        print(f"⛔ Skipping {sheet_name}: missing 'event_number' or 'transcript'")
        continue

    results = []
    for _, row in df.iterrows():
        if pd.isna(row[event_col]):
            continue

        text = str(row[text_col]) if pd.notnull(row[text_col]) else ""
        text = text.strip()

        if len(text) == 0:
            results.append({
                'event_number': int(row[event_col]),
                'event_text': "",
                'predicted_label': 'neutral',
                'confidence': 1.0,
                'pos_score': 0.0,
                'neu_score': 1.0,
                'neg_score': 0.0
            })
            continue

        try:
            # Use token-based truncation instead of character-based
            outputs = classifier(text, truncation=True, max_length=512, top_k=None)
            
            # Convert labels to human-readable format and build score dictionary
            score_dict = {}
            for result in outputs:
                readable_label = LABEL_MAPPING.get(result['label'], result['label']).lower()
                score_dict[readable_label] = result['score']
            
            # Get the top result with human-readable label
            top_result = max(outputs, key=lambda x: x['score'])
            readable_top_label = LABEL_MAPPING.get(top_result['label'], top_result['label']).lower()

            results.append({
                'event_number': int(row[event_col]),
                'event_text': text,
                'predicted_label': readable_top_label,
                'confidence': top_result['score'],
                'pos_score': score_dict.get('positive', 0.0),
                'neu_score': score_dict.get('neutral', 0.0),
                'neg_score': score_dict.get('negative', 0.0),
                'valence_event_continuous': (score_dict.get('positive', 0.0)) - (score_dict.get('negative', 0.0)),
                'arousal_event_score': 1 - score_dict.get('neutral', 0.0)
            })

        except Exception as e:
            print(f"⚠️ Error in {sheet_name}, event {row[event_col]}: {e}")
            continue

    # Save result
    out_df = pd.DataFrame(results)
    out_df.to_csv(os.path.join(save_dir, f"{sheet_name}_sentiment_transformer.csv"), index=False)

print("✅ Finished sentiment analysis for all sheets.")


# analyzer = SentimentIntensityAnalyzer()
# xls = pd.ExcelFile(excel_path)

# for sheet_name in xls.sheet_names:
#     df = pd.read_excel(xls, sheet_name=sheet_name)

#     if 'event_number' not in df.columns or 'Transcript' not in df.columns:
#         print(f"Sheet {sheet_name} skipped (missing columns).")
#         continue

#     results = []
#     for _, row in df.iterrows():
#         if pd.isna(row['event_number']):
#             continue

#         text = str(row['Transcript']) if pd.notnull(row['Transcript']) else ""
#         scores = analyzer.polarity_scores(text)

#         results.append({
#             'event_number': int(row['event_number']),
#             'event_text': text,
#             'neg': scores['neg'],
#             'neu': scores['neu'],
#             'pos': scores['pos'],
#             'compound': scores['compound']
#         })

#     result_df = pd.DataFrame(results)
#     result_df.to_csv(os.path.join(save_dir, f"{sheet_name}_sentiment_vader.csv"), index=False)

# print("✅ VADER sentiment saved to:", save_dir)
