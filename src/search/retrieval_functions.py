from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb


def bm25_retrieval(documents: List[str], query: str, top_k: int = 10):
    vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            token_pattern=r'\b\w+\b'
        )
    doc_vectors = vectorizer.fit_transform(documents)

    query_vector = vectorizer.transform([query])

    doc_lengths = [len(doc.split()) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    
    # Calculate BM25 scores
    scores = []
    for i, doc_vector in enumerate(doc_vectors):
        score = _calculate_bm25_score(documents, avg_doc_length, query_vector, doc_vector, i)
        scores.append((i, score))
        
    # Sort by score and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def _calculate_bm25_score(documents, avg_doc_length, query_vector, doc_vector, doc_idx: int) -> float:
        """Calculate BM25 score for query-document pair"""
        k1 = 1.2
        b = 0.75
        doc_length = len(documents[doc_idx].split())
        length_norm = k1 * ((1 - b) + b * (doc_length / avg_doc_length))
        
        # Simplified BM25 calculation using TF-IDF vectors
        tf_scores = doc_vector.toarray()[0]
        query_terms = query_vector.toarray()[0]
        
        bm25_score = 0
        for term_idx, query_weight in enumerate(query_terms):
            if query_weight > 0:
                tf = tf_scores[term_idx]
                bm25_score += query_weight * (tf * (k1 + 1)) / (tf + length_norm)
        
        return bm25_score

def dense_retrieval(document_embeddings: np.ndarray, 
                    query: str, top_k: int = 10, 
                    model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[int, float]]:
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = np.dot(query_embedding, document_embeddings.T)
    similarities = similarities / (
        np.linalg.norm(query_embedding) * 
        np.linalg.norm(document_embeddings, axis=1)
    )
            
    # Get top-k results
    top_indices = np.argsort(similarities[0])[::-1][:top_k]
    results = [(idx.item(), similarities[0][idx]) for idx in top_indices]
            
    return results

def reciprocal_rank_fusion(
        results_list: List[List[Tuple[int, float]]], 
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """
        Combine ranked lists using Reciprocal Rank Fusion
        RRF score = sum(1 / (rank + k)) for each retrieval method
        """
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc_id, score) in enumerate(results):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += 1 / (rank + k)
        
        # Sort by combined score
        sorted_results = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(doc_id, score) for doc_id, score in sorted_results]


db_path = "chroma_data/"
embed_model = "all-MiniLM-L6-v2"
collection_name = "f1_races"

client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(collection_name)

data = collection.get(include=['embeddings', 'documents'])
embeddings = data['embeddings']
documents = data['documents']

queries = [
     'Who won the 2013 Monaco Grand Prix?', 
     'Where was the 1985 Italian Grand Prix held?',
     'Who finished second in 2013 Abu Dhabi Grand Prix',
     'What is the official name of the 2021 Mexico City Grand Prix?']

for q in queries:
    print(f"Search querie: {q}") 
    dense_results = dense_retrieval(embeddings, q)
    print(f"Results for the dense search: {dense_results}\n")

    bm25_results = bm25_retrieval(documents, q)
    print(f"Results for the bm25 retrieval: {bm25_results}\n")

    fusion_results = reciprocal_rank_fusion([dense_results, bm25_results])
    print(f"Results for the fusion: {fusion_results}\n")

    dense_ids = [d[0] for d in dense_results]
    bm25_ids = [b[0] for b in bm25_results]
    print(f"Shared dense-bm25: {[id for id in dense_ids if id in bm25_ids]}")
    print('\n\n')