"""
Document Ingestion Script

This script processes the Constitution PDF and creates a searchable vector database.

What does this script do?
1. Loads the PDF file
2. Splits it into smaller chunks (so we can search efficiently)
3. Converts text chunks into embeddings (numerical representations)
4. Saves everything to a FAISS vector store (for fast similarity search)

Run this script ONCE before using main.py to create the vector database.
"""

import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

def ingest_documents():
    """
    Main function to process the PDF and create the vector store.
    """
    
    print("=" * 60)
    print("Starting Document Ingestion Process")
    print("=" * 60)
    
    # Step 1: Load the PDF Document
    # PyPDFLoader reads the PDF file and extracts all text from it
    # It returns a list of Document objects, where each page is a separate document
    print(f"\n[Step 1/4] Loading PDF from: {config.PDF_PATH}")
    
    # Check if PDF file exists
    if not os.path.exists(config.PDF_PATH):
        raise FileNotFoundError(
            f"PDF file not found at {config.PDF_PATH}. "
            "Please make sure the file exists."
        )
    
    # Load the PDF
    # This reads all pages from the PDF and extracts text
    loader = PyPDFLoader(config.PDF_PATH)
    all_documents = loader.load()
    
    # Skip the first 19 pages (Table of Contents/Index)
    # Pages 0-18 are the index, we start from page 19 (index 19) which is page 20
    # This ensures we only process actual Article content, not the table of contents
    SKIP_PAGES = 19
    documents = all_documents[SKIP_PAGES:]
    
    print(f"[OK] Loaded {len(all_documents)} total pages, skipping first {SKIP_PAGES} pages (Table of Contents)")
    print(f"[OK] Processing {len(documents)} pages starting from page {SKIP_PAGES + 1}")
    
    # Clean the document content before chunking
    # PDFs often have formatting issues where sentences are broken by excessive newlines
    # This cleaning step replaces multiple newlines with spaces to fix broken sentences
    # This prevents Article headers from being separated from their content
    print("\n[Step 1.5/4] Cleaning document content (fixing broken sentences)...")
    
    for doc in documents:
        # Replace multiple newlines (2 or more) with a single space
        # This fixes sentences that are broken across lines in the PDF
        # We keep single newlines as they might be intentional paragraph breaks
        cleaned_content = re.sub(r'\n{2,}', ' ', doc.page_content)
        # Replace single newlines that are not followed by capital letters (likely mid-sentence breaks)
        # This is a heuristic: if a newline is followed by lowercase, it's probably a broken sentence
        cleaned_content = re.sub(r'\n(?=[a-z])', ' ', cleaned_content)
        # Clean up any remaining excessive whitespace
        cleaned_content = re.sub(r' +', ' ', cleaned_content)
        doc.page_content = cleaned_content.strip()
    
    print("[OK] Document content cleaned (excessive newlines replaced with spaces)")
    
    # Step 2: Split Documents into Chunks
    # Why do we need chunking?
    # - LLMs have token limits (can't process entire document at once)
    # - Smaller chunks allow precise retrieval (find exact relevant sections)
    # - Makes search more efficient (compare smaller pieces)
    # 
    # What is RecursiveCharacterTextSplitter?
    # - It tries to split text at natural boundaries (paragraphs, sentences, words)
    # - "Recursive" means it tries different splitting strategies in order
    # - First tries paragraphs, then sentences, then words, then characters
    # - This preserves meaning better than just splitting at fixed character counts
    print(f"\n[Step 2/4] Splitting documents into chunks...")
    print(f"  Chunk size: {config.CHUNK_SIZE} characters")
    print(f"  Overlap: {config.CHUNK_OVERLAP} characters")
    
    # Create the text splitter with our configuration
    # RecursiveCharacterTextSplitter tries to split at natural boundaries
    # It tries: paragraphs -> sentences -> words -> characters
    # This preserves meaning better than fixed character splits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,      # Maximum size of each chunk (2500 chars)
        chunk_overlap=config.CHUNK_OVERLAP, # How much chunks overlap (400 chars)
        length_function=len,                # Function to measure text length
        separators=["\n\n", "\n", ". ", " ", ""]  # Try to split at these boundaries in order
    )
    
    # Split all documents into chunks
    # This takes our list of page documents and breaks them into smaller chunks
    chunks = text_splitter.split_documents(documents)
    
    print(f"[OK] Created {len(chunks)} text chunks from the document")
    
    # Step 3: Create Embeddings
    # What are Embeddings?
    # - Embeddings are numerical representations of text (vectors/arrays of numbers)
    # - Similar texts have similar embeddings (close in vector space)
    # - Example: "Constitution" and "Legal document" would have similar embeddings
    # - This allows us to find semantically similar text, not just exact matches
    # 
    # How do embeddings work?
    # - OpenAI's embedding model converts text into a 1536-dimensional vector
    # - Each number in the vector captures some aspect of the text's meaning
    # - We can then calculate "distance" between vectors to find similar texts
    print(f"\n[Step 3/4] Creating embeddings using OpenAI...")
    print("  This converts text chunks into numerical vectors for similarity search")
    
    # Create the embedding model
    # OpenAIEmbeddings uses OpenAI's text-embedding-ada-002 model by default
    # This model converts text into 1536-dimensional vectors
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Step 4: Create and Save Vector Store
    # What is a Vector Store?
    # - A database optimized for storing and searching embeddings
    # - FAISS (Facebook AI Similarity Search) is very fast at finding similar vectors
    # - It uses indexing techniques to search millions of vectors in milliseconds
    # 
    # How does it work?
    # - We store each chunk's text and its embedding together
    # - When searching, we convert the query to an embedding
    # - FAISS finds the most similar chunk embeddings to the query embedding
    # - Returns the actual text chunks that are most relevant
    print(f"\n[Step 4/4] Creating FAISS vector store...")
    
    # Create the vector store from chunks
    # from_documents does two things:
    # 1. Converts all chunks to embeddings (calls OpenAI API)
    # 2. Stores them in a FAISS index for fast retrieval
    vector_store = FAISS.from_documents(
        documents=chunks,      # Our text chunks
        embedding=embeddings   # The embedding model to use
    )
    
    # Save the vector store to disk
    # This creates a folder with index files that we can load later
    # We don't need to reprocess the PDF every time - just load this index
    print(f"\nSaving vector store to: {config.VECTOR_DB_PATH}")
    vector_store.save_local(config.VECTOR_DB_PATH)
    
    print("\n" + "=" * 60)
    print("[OK] Document ingestion completed successfully!")
    print("=" * 60)
    print(f"\nYou can now run main.py to start asking questions.")
    print(f"Vector store saved at: {config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    # This runs the ingestion process when you execute: python ingest.py
    try:
        ingest_documents()
    except Exception as e:
        print(f"\n[ERROR] Error during ingestion: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your OPENAI_API_KEY is set correctly in .env file")
        print("2. Verify that the PDF file exists at the specified path")
        print("3. Make sure you have an active internet connection")
        raise

