from datasets import load_from_disk
from unidecode import unidecode
import json

ds = load_from_disk("D:/University/trsshi/filtered_dataset")
ordered = ds.sort("title")
processed = []
for doc in ordered:
    cleanedDoc = {
        "id": doc["id"],
        "url": doc["url"],
        "title": unidecode(doc["title"]),
        "text": unidecode(doc["text"])
    }

    if cleanedDoc["title"][:4].isdigit() and int(cleanedDoc["title"][:4]) < 1950:
        continue
    
    text_sections = cleanedDoc["text"].split("\n\n")
    correct_sections = []
    for section in text_sections:
        if str.lower(section).strip() == "references":
            break
        if len(section.split()) > 15:
            correct_sections.append(section)
    
    cleanedDoc["text"] = (" ".join(correct_sections)).replace("\n", " ")
    
    if len(cleanedDoc["text"]) > 100:
        processed.append(cleanedDoc)

with open('articles_cleaned.jsonl', 'w', encoding='utf-8') as f:
    for record in processed:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')