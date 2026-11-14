import json
import chromadb
from chromadb.utils import embedding_functions

db_path = "chroma_data/"
embed_model = "all-MiniLM-L6-v2"
collection_name = "f1_races"

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(
    collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model
    ))


eval_sets = []
found_in_5 = 0
found_in_10 = 0

with open('tests/eval-set.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: 
            eval_sets.append(json.loads(line))

for set in eval_sets:
    print(f"Quering '{set['request']}'")
    query_results = collection.query(
            query_texts=[set["request"]],
            n_results=10
        )

    results_ids = query_results['ids'][0]

    if set["chunk"] in results_ids:
        found_in_10 += 1
        print("Found in 10!")

        if set["chunk"] in results_ids[:5]:
            found_in_5 += 1
            print("Found in 5!")
    else:
        print("Correct Id not found")
        print(f"Returned Ids: {results_ids}")

print()
print("Recall@5: " + str(found_in_5 / len(eval_sets)))
print("Recall@5: " + str(found_in_10 / len(eval_sets)))