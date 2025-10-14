from eyelinkparser.parser import EdfParser
import pandas as pd
import os

# === Set paths ===
edf_path = '/Users/UChicago/CASNL/storyfest/data/pupil/1_raw/encoding/1001_storyfest_encoding.EDF'
output_csv_path = '/Users/UChicago/CASNL/storyfest/data/pupil/1_csv/encoding/1001_samples.csv'
output_event_path = '/Users/UChicago/CASNL/storyfest/data/pupil/1_csv/encoding/1001_events.csv'
# === Parse EDF ===
print("Parsing EDF file...")
parser = EdfParser(edf_path)
parser.parse_samples()
samples = parser.dm['samples']

# Convert to DataFrame
df = pd.DataFrame(samples)
print("Available columns:", df.columns.tolist())

# Save to CSV
os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
df.to_csv(output_csv_path, index=False)
print(f"Saved sample data to {output_csv_path}")

parser.parse_events()
events = parser.dm['events']
events_df = pd.DataFrame(events)
events_df.to_csv(output_event_path, index=False)
print(f"Saved event data to {output_event_path}")
