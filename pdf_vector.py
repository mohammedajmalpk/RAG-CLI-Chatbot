import pickle
import numpy as np
import faiss
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_INDEX_FILE = "vector_store/vectors.index"
CHUNK_FILE = "vector_store/chunks.pkl"
CHUNK_SIZE = 500  # goldilocks principle, proven by many RAG models, as per their white paper docs
OVERLAP_SIZE = 100  # 20% of chunk size

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def read_pdf(file_path):
    """
        read a introduction to python pdf file and return concatenated text and page wise info.
        2-data ingestion
    """
    print("\nReading pdf...\n")
    with open(file_path, "rb") as f:
        pdf_reader = PdfReader(f)
        total_pages = len(pdf_reader.pages)
        page_texts = []

        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                "text": page_text,
                "page_num": page_num + 1
            })

        text = ''.join(p["text"] for p in page_texts)

    print(f"Total pages: {total_pages}")
    print(f"Total text length: {len(text)}")
    print(f"Average characters per page: {len(text)//total_pages}")

    return text, total_pages


def create_chunks(text, total_pages, chunk_size=CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    """
        split the text into overlapping chunks and create chunk metadata.
        3- data parsing
    """
    chunks = []
    chunk_metadata = []
    step_size = chunk_size - overlap_size  # here 500-100 = 400

    for chunk_start_pos in range(0, len(text), step_size):
        chunk_text = text[chunk_start_pos:chunk_start_pos + chunk_size]
        chunks.append(chunk_text)

        per_page_text = len(text) // total_pages
        estimated_page = min((chunk_start_pos // per_page_text) + 1, total_pages)

        chunk_metadata.append({
            "start_pos": chunk_start_pos,
            "estimated_page": estimated_page
        })

    return chunks, chunk_metadata


def embed_chunks(chunks):
    """
        convert each chunk into a vector using the SentenceTransformer model.
        4-embedding
    """
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)}")
        vector = model.encode(chunk)
        embeddings.append(vector)

    embeddings = np.array(embeddings).astype('float32')
    return embeddings


def build_faiss_index(embeddings):
    """
        Build a faiss index from embeddings. here we are using hugging face
        5 - create vector database
    """
    dim = embeddings.shape[1] # so the vector dimension is 384
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_data(index, chunks, chunk_metadata, total_pages):
    """
        save FAISS index and chunk data to files.
        6 - store embeddings to vector db
    """
    faiss.write_index(index, VECTOR_INDEX_FILE)
    with open(CHUNK_FILE, "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'chunk_metadata': chunk_metadata,
            'total_pages': total_pages
        }, f)

    print("Vector database created successfully!")
    print(f"Files saved: {VECTOR_INDEX_FILE}, {CHUNK_FILE}")


def run_pdf_reader(user_pdf_file):
    """
        read pdf
        create chunk based on chunk size 500 (Goldilocks principle)
        embed each chunk for convert to vector
        create a vector db and all embeddings added to vector db
        save the vector db and chunks pickle file

        1- main function
    """
    text, total_pages = read_pdf(user_pdf_file)
    chunks, chunk_metadata = create_chunks(text, total_pages) # this is data parsing
    embeddings = embed_chunks(chunks) # this is embedding
    index = build_faiss_index(embeddings) # this is storing vectors
    save_data(index, chunks, chunk_metadata, total_pages) # this is save

    print(f"embeddings shape: {embeddings.shape}")
    print(f"first 5 vector dimensions: {embeddings[0][:5]}")

    return embeddings, chunks


if __name__ == "__main__":
    pdf_file = "pdf-folder/Introduction-To-Python.pdf"
    embeddings, chunks = run_pdf_reader(pdf_file)
    print("\nsetup complete! You may start to chat...\n")
