import json

records = []
chunks = []
chunk_id = 1
default_chunk_size = 350
with open('articles_checked.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: 
            records.append(json.loads(line))

for record in records:
    text = str(record["text"])
    text_words = text.split()
    if len(text_words) < 500:
        chunks.append({
            "chunk_id": chunk_id,
            "article_id": record["id"],
            "chunk_in_article": 0,
            "chunk_text": record["text"],
            "chunk_size_words": len(text_words),
            "chunk_size_symbols": len(text)
        })
        chunk_id += 1
    else:
        chunks_count = int(len(text_words) / default_chunk_size)
        text_chunk_size = int(len(text_words) / chunks_count)
        start_point = 0
        for i in range(chunks_count):
            end_point = start_point + text_chunk_size
            if end_point < len(text_words):
                while end_point < len(text_words) and not (text_words[end_point].endswith('.')
                                        or text_words[end_point].endswith('!')
                                        or text_words[end_point].endswith('?')):
                            end_point += 1
                end_point += 1        
                            
            else : 
                end_point = len(text_words)

            chunk_text = " ".join(text_words[start_point:end_point])

            if i > 0:
                chunk_text = "The following information is related to " + record["title"] + ". " + chunk_text 

            chunks.append({
                        "chunk_id": chunk_id,
                        "article_id": record["id"],
                        "chunk_in_article": i,
                        "chunk_text": chunk_text,
                        "chunk_size_words": len(chunk_text.split()),
                        "chunk_size_symbols": len(chunk_text)
                    })
            
            chunk_id += 1

            start_point = int(end_point - text_chunk_size * 0.25)
            while start_point < len(text_words) and not (text_words[start_point].endswith('.')
                                   or text_words[start_point].endswith('!')
                                   or text_words[start_point].endswith('?')):
                start_point += 1
            start_point += 1
            
with open('data_chunks.jsonl', 'w', encoding='utf-8') as f:
    for record in chunks:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
