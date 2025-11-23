import pandas as pd
import json

# --- CONFIG ---
xlsx_file = "../data/Training_Set.xlsx"  # path to your XLSX
output_file = "../data/train.jsonl"      # output JSONL for training

# Map raw dataset entity types to required labels
ENTITY_MAP = {
    "Name": "PERSON_NAME",
    "Credit Card": "CREDIT_CARD",
    "Email": "EMAIL",
    "Phone": "PHONE",
    "Address": "LOCATION",
    "City": "CITY",
    "Company": "LOCATION",
    "SSN": None,   # ignore
    "URL": None,   # ignore
    "Date": "DATE"
}

# Only keep these labels
REQUIRED_LABELS = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"}

# Load Excel file
df = pd.read_excel(xlsx_file)

jsonl_rows = []

for idx, row in df.iterrows():
    text = row["Text"]
    entities = []

    # Get True Predictions column
    true_preds = row.get("True Predictions")
    
    # Convert string to list if stored as string
    if isinstance(true_preds, str):
        try:
            true_preds = eval(true_preds)
        except:
            true_preds = []

    # Map and filter entities
    for start, end, label in true_preds:
        mapped_label = ENTITY_MAP.get(label.capitalize(), None)
        if mapped_label and mapped_label in REQUIRED_LABELS:
            entities.append({
                "start": start,
                "end": end,
                "label": mapped_label
            })

    # Only include row if it has at least one required entity
    if entities:
        jsonl_rows.append({
            "id": f"utt_{idx+1:04d}",
            "text": text,
            "entities": entities
        })

# Write to JSONL
with open(output_file, "w", encoding="utf-8") as f:
    for row in jsonl_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Converted {len(jsonl_rows)} rows to {output_file}")
