OUTPUT_FORMAT = """ The response must be given in JSON format as follows:
{
    "chunks_used": [id1, id2, ..., idn],
    "response": "...."
}

Here the chunks_used must be a list of ids that the data fragments were presented with. Only add the chunks that contained valuable information. If the answer to the question was not found, list the ids of all the chunks used.
The response should be a string, and it should be short and precise. If the question is open - provide 2-3 main points you can find.
"""

PROMPT = """Answer the following question given the information bellow: {question}

Use this information to answer the question correctly:
{chunk1_id}: {chunk1}
{chunk2_id}: {chunk2}
{chunk3_id}: {chunk3}
{chunk4_id}: {chunk3}
{chunk5_id}: {chunk4}

Here the first number is the chunk_id, and the following text is the actual text data that you can use to answer the questions.

{output_format}

If the answer to the question is not found in the text fragments you are givent, you must state so openly in the response. Example: "I apologize, but I was unable to find the answer to your question".
"""

from pydantic import BaseModel
import numpy as np
from typing import List, Tuple

class ResponseModel(BaseModel):
    chunks_used: list[int]
    response: str

embeddings = []
documents = []

avg_doc_length = 0

def answer(question: str):
    from chromadb import Knn, Rrf, Search
    import llm
    global embeddings
    global documents

    if (avg_doc_length == 0 or len(documents) == 0 or len(embeddings) == 0):
        calculate_initial_data()

    dense_results = dense_retrieval(question)
    bm25_results = bm25_retrieval(question)
    fusion_results = reciprocal_rank_fusion([dense_results, bm25_results])

    ids = [d[0] for d in fusion_results[:5]]

    prompt = PROMPT.format(question=question, output_format=OUTPUT_FORMAT, 
                           chunk1_id=(ids[0]+1), chunk1=documents[ids[0]],
                           chunk2_id=(ids[1]+1), chunk2=documents[ids[1]],
                           chunk3_id=(ids[2]+1), chunk3=documents[ids[2]],
                           chunk4_id=(ids[3]+1), chunk4=documents[ids[3]],
                           chunk5_id=(ids[4]+1), chunk5=documents[ids[4]],)

    parsed, raw = llm.get_response(prompt, ResponseModel)
    for id in parsed.chunks_used:
         if not id - 1 in ids:
            return parsed.response, False 
    return parsed.response, True

def calculate_initial_data():
    import chromadb
    global avg_doc_length
    global embeddings
    global documents

    db_path = "chroma_data/"
    collection_name = "f1_races"
    
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)
    
    data = collection.get(include=['embeddings', 'documents'])
    embeddings = data['embeddings']
    documents = data['documents']

    doc_lengths = [len(doc.split()) for doc in documents]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)


def bm25_retrieval(query, top_k = 5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    global avg_doc_length

    vectorizer = TfidfVectorizer(
                    lowercase=True,
                    stop_words='english',
                    token_pattern=r'\b\w+\b'
                )
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    
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

def dense_retrieval(query: str, top_k: int = 5, 
                    model_name: str = "all-MiniLM-L6-v2") -> List[Tuple[int, float]]:
    from sentence_transformers import SentenceTransformer
    global embeddings

    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])

    # Calculate cosine similarity
    similarities = np.dot(query_embedding, embeddings.T)
    similarities = similarities / (
        np.linalg.norm(query_embedding) * 
        np.linalg.norm(embeddings, axis=1)
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