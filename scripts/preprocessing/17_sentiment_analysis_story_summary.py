import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load transformer model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None, device=-1)

# Label mapping for human-readable labels
LABEL_MAPPING = {
    'LABEL_0': 'negative',
    'LABEL_1': 'neutral',
    'LABEL_2': 'positive'
}

STORY_VALENCE = {
    'Pool Party': 'positive',
    'Sea Ice': 'neutral',
    'Natalie Wood': 'negative',
    'Impatient Billionaire': 'positive',
    'Grandfather Clocks': 'neutral',
    'Dont Look': 'negative'
}

# Load Excel file
# --- Setup ---
save_dir = '/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/17_sentiment_by_story_transformer'  # Updated directory
os.makedirs(save_dir, exist_ok=True)
file_path = "/Users/UChicago/CASNL/storyfest/experiment/Transcript_Summary.xlsx"
df = pd.read_excel(file_path)

# Display column names to find the text column
print(df.columns)

results = []

for _, row in df.iterrows():
    story_name = str(row['Story'])
    text = str(row['Transcript_Summary']).strip()

    if not text or text.lower() == "nan":
        print(f"⚠️ Skipping {story_name}: empty summary")
        continue

    try:
        # Run transformer sentiment
        outputs = classifier(text, truncation=True, max_length=512, top_k=None)

        # Map labels and extract scores
        score_dict = {}
        for result in outputs:
            readable_label = LABEL_MAPPING.get(result['label'], result['label']).lower()
            score_dict[readable_label] = result['score']

        top_result = max(outputs, key=lambda x: x['score'])
        top_label = LABEL_MAPPING.get(top_result['label'], top_result['label']).lower()

        results.append({
            'story': story_name,
            'valence_label': STORY_VALENCE[story_name],
            'predicted_label': top_label,
            'confidence': top_result['score'],
            'pos_score': score_dict.get('positive', 0.0),
            'neu_score': score_dict.get('neutral', 0.0),
            'neg_score': score_dict.get('negative', 0.0),
            'valence_story_continuous': score_dict.get('positive', 0.0) - score_dict.get('negative', 0.0)
        })

    except Exception as e:
        print(f"❌ Error processing {story_name}: {e}")

# Save all results to one CSV
out_df = pd.DataFrame(results)
out_df.to_csv(os.path.join(save_dir, 'all_story_sentiment_summary.csv'), index=False)
print("✅ Done! Saved sentiment scores for all stories.")