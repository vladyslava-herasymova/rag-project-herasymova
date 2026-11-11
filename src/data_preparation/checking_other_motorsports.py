import json

count = 0
f1_data = []
with open('articles_cleaned.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        
        if line: 
            record = json.loads(line)
            if ("macau grand prix" in str.lower(record["title"])
                or "indycar" in str.lower(record["title"])
                or "echo park" in str.lower(record["title"])
                or "texas" in str.lower(record["title"])
                or "generac" in str.lower(record["title"])
                or "music city" in str.lower(record["title"])):
                print(record["id"] + ': ' + record["url"] + "   " + record["title"])
                count += 1
            else:
                f1_data.append(record)
        
with open('articles_checked.jsonl', 'w', encoding='utf-8') as f:
    for record in f1_data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')