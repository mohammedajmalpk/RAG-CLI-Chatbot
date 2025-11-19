import os
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer
import faiss


FAISS_INDEX = "vector_store/vectors.index"
CHUNK_FILE = "vector_store/chunks.pkl"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def check_db_exists():
    """
        check if FAISS index and chunk data exist. return True if both exist, else False.
    """
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(CHUNK_FILE):
        return False
    return True


def load_faiss_index():
    """
        load the FAISS index from file.
    """
    print("Loading FAISS index...")
    return faiss.read_index(FAISS_INDEX)


def load_chunks():
    """
        load chunks and chunk_metadata from pickle file.
    """
    print("Loading chunk data...")
    with open(CHUNK_FILE, "rb") as f:
        data = pickle.load(f)
    return data['chunks'], data['chunk_metadata'], data['total_pages']


def vector_search(user_query, index, chunks, chunk_metadata, top_k=3):
    """
        convert user query to vector, search FAISS index, and return top matching chunks.
    """
    query_vector = embedding_model.encode(user_query)
    query_vector = np.array([query_vector]).astype('float32')

    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "score": float(dist),
            "chunk": chunks[idx],
            "page": chunk_metadata[idx]["estimated_page"]
        })

    return results


def generate_answer(user_query, matched_chunks):
    """
        Generate a final answer combining user query and retrieved chunks.
        use ollama model as transformer LLM
    """
    context = "\n\n".join([f"Page: {c['page']}\n{c['chunk']}\n" for c in matched_chunks])
    prompt = (
        f"You are a helpful assistant. Use the context below to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )
    response = ollama.generate(model='llama3.2', prompt=prompt)
    # response = f"Your question: {user_query}\n\nAnswer:\n{context}"
    return response['response']


def main():
    if not check_db_exists():
        print("FAISS index or chunk file not found. Load the PDF vector file again.")
        return

    # Load data
    index = load_faiss_index()
    chunks, chunk_metadata, total_pages = load_chunks()

    print(f"Loaded {len(chunks)} chunks from {total_pages} pages.")

    while True:
        query = input("\nThis is your space. Ask your question about introduction to python (or type 'exit'): ")
        if query.lower() in ["exit", "q", "quit"]:
            print("Thank you!")
            break

        matches = vector_search(query, index, chunks, chunk_metadata)
        print("\nTop 3 retrieved chunks:")
        for item in matches:
            print("\nScore:", item["score"])
            print("Page:", item["page"])
            print("Chunk:", item["chunk"][:500], "...")  # show first 500 chars

        final_response = generate_answer(query, matches)
        print("\n==========================")
        print(final_response)
        print("==========================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", str(e))
