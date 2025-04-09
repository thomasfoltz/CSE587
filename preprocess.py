import csv
import json
import re

class_labels = {
    1: "world",
    2: "sports",
    3: "business",
    4: "sci/tech"
}

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return text

def preprocess_csv_to_json(input_csv, output_json):
    data = []
    with open(input_csv, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            label = class_labels.get(int(row["Class Index"]), "unknown")
            title = row["Title"].strip()
            description = row["Description"].strip()
            text = f"{title}. {description}"
            
            data.append({
                "text": clean_text(text),
                "label": label
            })
    
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)

preprocess_csv_to_json("train.csv", "train.json")
preprocess_csv_to_json("test.csv", "val.json")