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

while True:
    print("Please, enter your request. Type Enter to quit.")
    request = input()
    if len(request) == 0:
        break

    query_results = collection.query(
        query_texts=[request],
        n_results=5,
        include=["documents", "distances"]
    )

    print("Here are the results for your request:")
    for i in range (5):
        print(f"Result {i + 1}:")
        print(f"Id: {query_results['ids'][0][i]}")
        print(f"Text: {query_results['documents'][0][i]}")
        print(f"Distance: {query_results['distances'][0][i]}")

    print()
    print()
