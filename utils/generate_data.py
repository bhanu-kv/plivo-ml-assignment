import pandas as pd
import json
import os

train_xlsx = "data/Training_Set.xlsx"
test_xlsx = "data/Testing_Set.xlsx"
train_output_file = "data/train.jsonl"
dev_output_file = "data/dev.jsonl"
test_output_file = "data/test.jsonl"

# Map raw entity types to required labels
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

REQUIRED_LABELS = set(ENTITY_MAP.values()) - {None}

def process_excel(df, include_entities=True):
    """
    Convert a dataframe to JSONL rows suitable for train/dev/test.
    include_entities=True => produce 'entities' field
    """
    rows = []
    for idx, row in df.iterrows():
        text = row["Text"]
        entities = []
        if include_entities:
            true_preds = row.get("True Predictions", [])
            if isinstance(true_preds, str):
                try:
                    true_preds = eval(true_preds)
                except:
                    true_preds = []

            for start, end, label in true_preds:
                mapped_label = ENTITY_MAP.get(label.capitalize(), None)
                if mapped_label and mapped_label in REQUIRED_LABELS:
                    entities.append({
                        "start": int(start),
                        "end": int(end),
                        "label": mapped_label
                    })

        obj = {
            "id": f"utt_{idx+1:04d}",
            "text": text
        }
        if include_entities and entities:
            obj["entities"] = entities

        if include_entities:
            if entities:
                rows.append(obj)
        else:
            rows.append(obj)

    return rows

train_df = pd.read_excel(train_xlsx)
train_rows = process_excel(train_df, include_entities=True)
with open(train_output_file, "w", encoding="utf-8") as f:
    for row in train_rows:
        f.write(json.dumps(row) + "\n")
print(f"Train JSONL saved: {len(train_rows)} rows -> {train_output_file}")

# Dev set: keep entities for evaluation
dev_rows = process_excel(train_df, include_entities=True)
with open(dev_output_file, "w", encoding="utf-8") as f:
    for row in dev_rows:
        f.write(json.dumps(row) + "\n")
print(f"Dev JSONL saved: {len(dev_rows)} rows -> {dev_output_file}")

# Test set: text only, no entities
test_df = pd.read_excel(test_xlsx)
test_rows = process_excel(test_df, include_entities=False)
with open(test_output_file, "w", encoding="utf-8") as f:
    for row in test_rows:
        f.write(json.dumps(row) + "\n")
print(f"Test JSONL saved: {len(test_rows)} rows -> {test_output_file}")
