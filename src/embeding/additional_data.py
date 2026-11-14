from sentence_transformers import SentenceTransformer
import json

embed_model = "all-MiniLM-L6-v2"

chunks = []
with open('data_chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: 
            chunks.append(json.loads(line))


model = SentenceTransformer('sentence-transformers/' + embed_model)
calculated_embeddings = model.encode(list(ch["chunk_text"] for ch in chunks[:1]))
print(len(calculated_embeddings[0]))
print(len(chunks))