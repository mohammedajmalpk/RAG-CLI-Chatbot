📘 RAG-CLI-Chatbot

A CLI-based Retrieval-Augmented Generation (RAG) chatbot that reads PDFs, creates embeddings using HuggingFace Sentence-Transformers, stores vectors in FAISS, and answers user questions using semantic search. Ollama LLM (llama3.2) is used as the transformer for generation.

 - 🚀 Features 
 - 📄 Extract text from PDF files 
 - ✂️ Smart text chunking 
 - 🔍 Create embeddings using all-MiniLM-L6-v2 (HuggingFace)
 - 🧠 Store vectors in FAISS (CPU-friendly)
 - ❓ Ask queries and retrieve top matching chunks 
 - 🤖 Generate answers using Ollama LLM (llama3.2)
 - 💻 Fully CLI-based (no GUI required)

⚡ How It Works

 - Load PDFs → extract raw text. 
 - Chunk text → divide into smaller, meaningful pieces. 
 - Generate embeddings → convert chunks into vector representations. 
 - Store in FAISS → fast vector search. 
 - Query system → retrieve top matching chunks using semantic search. 
 - Answer generation → Ollama LLM generates context-aware answers.

🛠️ Requirements

 - Python 3.11+ 
 - Ollama - installed with llama3.2 model (locally run)
 - HuggingFace sentence-transformers 
 - FAISS (faiss-cpu)
 - PyPDF2 (for PDF text extraction)