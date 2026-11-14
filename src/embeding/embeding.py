import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import json

db_path = "chroma_db/"
embed_model = "all-MiniLM-L6-v2"
collection_name = "f1_races"

chunks = []
with open('data_chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: 
            chunks.append(json.loads(line))


model = SentenceTransformer('sentence-transformers/' + embed_model)
calculated_embeddings = model.encode(list(ch["chunk_text"] for ch in chunks))
print(calculated_embeddings[0])

client = chromadb.PersistentClient(path=db_path)

collection = client.create_collection(
    collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    )
)

embeddings_list = list(calculated_embeddings.tolist())

collection.add(
    ids=[str(ch["chunk_id"]) for ch in chunks],
    documents=[ch["chunk_text"] for ch in chunks],
    embeddings=embeddings_list,
    metadatas=[{
        "article_id": ch["article_id"],
        "chunk_in_article": ch["chunk_in_article"]}
        for ch in chunks],
)

