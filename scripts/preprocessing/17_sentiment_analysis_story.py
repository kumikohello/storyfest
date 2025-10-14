import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load transformer model
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, top_k=None, device=0 if torch.cuda.is_available() else -1)

# Label mapping
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

# --- Setup ---
save_dir = '/Users/UChicago/CASNL/storyfest/data/pupil/3_processed/17_sentiment_story'
os.makedirs(save_dir, exist_ok=True)
file_path = "/Users/UChicago/CASNL/storyfest/experiment/all_story_transcript.xlsx"
df = pd.read_excel(file_path)

results = []

def chunk_text(text, tokenizer, max_length=512, stride=256):
    """Split text into overlapping chunks of token length <= max_length."""
    tokens = tokenizer(text, return_tensors='pt', truncation=False)['input_ids'][0]
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+max_length]
        if len(chunk) == 0:
            continue
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        if i + max_length >= len(tokens):
            break
    return chunks

for _, row in df.iterrows():
    story_name = str(row['Story'])
    text = str(row['Transcript']).strip()

    if not text or text.lower() == "nan":
        print(f"⚠️ Skipping {story_name}: empty summary")
        continue

    try:
        chunks = chunk_text(text, tokenizer)
        scores_accum = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
        counts = {'positive': 0, 'neutral': 0, 'negative': 0}

        for chunk in chunks:
            outputs = classifier(chunk, truncation=True, max_length=512, top_k=None)
            for result in outputs:
                label = LABEL_MAPPING.get(result['label'], result['label']).lower()
                scores_accum[label] += result['score']
                counts[label] += 1

        # Compute average scores
        avg_scores = {
            label: (scores_accum[label] / counts[label]) if counts[label] > 0 else 0.0
            for label in ['positive', 'neutral', 'negative']
        }
        top_label = max(avg_scores, key=avg_scores.get)

        results.append({
            'story': story_name,
            'valence_label': STORY_VALENCE[story_name],
            'predicted_label': top_label,
            'confidence': avg_scores[top_label],
            'pos_score': avg_scores['positive'],
            'neu_score': avg_scores['neutral'],
            'neg_score': avg_scores['negative'],
            'valence_story_continuous': avg_scores['positive'] - avg_scores['negative']
        })

    except Exception as e:
        print(f"❌ Error processing {story_name}: {e}")

# Save to CSV
out_df = pd.DataFrame(results)
out_df.to_csv(os.path.join(save_dir, 'all_story_sentiment.csv'), index=False)
print("✅ Done! Saved sentiment scores for all stories.")
