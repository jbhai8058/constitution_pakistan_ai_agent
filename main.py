"""
Main Chat Interface for Constitution Q&A Agent

This is the main script that provides an interactive chat interface
for asking questions about the Constitution of Pakistan.

How it works:
1. Loads the pre-built vector store (created by ingest.py)
2. Sets up a RetrievalQA chain (combines retrieval + question answering)
3. Waits for user questions and provides answers based on the Constitution

Run this script after running ingest.py to start asking questions.
"""

# Suppress Pydantic warnings for cleaner terminal output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.*")

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import config

def load_vector_store():
    """
    Load the FAISS vector store from disk.
    This was created by running ingest.py earlier.
    """
    print("Loading vector store from disk...")
    
    # Check if vector store exists
    if not os.path.exists(config.VECTOR_DB_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {config.VECTOR_DB_PATH}. "
            "Please run ingest.py first to create the vector store."
        )
    
    # Create embeddings model (same as used during ingestion)
    # We need this to convert the query text into an embedding for search
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Load the FAISS vector store from disk
    # This loads the index we created in ingest.py
    vector_store = FAISS.load_local(
        folder_path=config.VECTOR_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS loading
    )
    
    print("[OK] Vector store loaded successfully!")
    return vector_store

def create_qa_chain(vector_store):
    """
    Create a RetrievalQA chain that combines retrieval and question answering.
    
    What is a Chain?
    - A chain is a sequence of operations that work together
    - In RAG (Retrieval-Augmented Generation), the chain does:
      1. RETRIEVAL: Finds relevant chunks from the vector store
      2. AUGMENTATION: Adds those chunks as context to the prompt
      3. GENERATION: LLM generates answer using the context
    
    How RetrievalQA Chain Works:
    1. User asks a question (e.g., "What is Article 1?")
    2. Chain converts question to embedding and searches vector store
    3. Retrieves top-k most relevant chunks (default is 4)
    4. Combines question + retrieved chunks into a prompt
    5. Sends prompt to LLM (GPT-4o-mini)
    6. LLM generates answer based on the provided context
    7. Returns answer to user
    """
    
    print("Setting up the Q&A chain...")
    
    # Step 1: Create the LLM (Large Language Model)
    # ChatOpenAI is LangChain's wrapper for OpenAI's chat models
    # We're using GPT-4o-mini as specified in config
    llm = ChatOpenAI(
        model=config.MODEL_NAME,           # gpt-4o-mini
        temperature=config.TEMPERATURE,    # 0.0 for deterministic answers
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Step 2: Create a Custom Prompt Template
    # This defines how we format the prompt sent to the LLM
    # The template includes:
    # - {context}: The retrieved chunks from the vector store (will be formatted as string)
    # - {question}: The user's question
    
    # ChatPromptTemplate is the modern way to create prompts in LangChain 1.x
    # It uses a list of messages (system, human, etc.)
    # Concise system prompt for direct, brief answers
    system_prompt = """You are a Pakistan Constitution expert. Answer the user's question directly and briefly. Do not add prefixes like "Answer:" or "Response:" - just provide the answer directly. Do not provide extra background unless asked. If the user asks 'What is Article 6', just state the law of High Treason simply. Do not mention that you are an AI. If a footnote or later text says an article was 'omitted' or 'substituted', prioritize the amendment over the original text. If the answer is not in the document, state that you don't know."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext from Constitution:\n{context}"),
        ("human", "{question}")
    ])
    
    # Step 3: Create the Retriever
    # The retriever finds relevant chunks from the vector store based on the question
    # Increased k to 6 to give the LLM more context
    # This helps catch amendments that might appear in later chunks
    # More chunks = better chance of seeing both original text and amendment footnotes
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks for better context
    )
    
    # Step 4: Create the Q&A Chain using LCEL (LangChain Expression Language)
    # LCEL is the modern way to create chains in LangChain 1.x
    # The chain works like this:
    # 1. Takes question as input
    # 2. Retrieves relevant documents using retriever
    # 3. Formats documents into context string
    # 4. Passes context + question to LLM via prompt
    # 5. LLM generates answer
    # 6. Output parser extracts the text answer
    
    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the chain using LCEL pipe operator
    # This chains together: retriever -> format_docs -> prompt -> llm -> output_parser
    # The chain works like this:
    # 1. Takes question as input
    # 2. Retrieves relevant documents using retriever
    # 3. Formats documents into context string
    # 4. Passes context + question to LLM via prompt
    # 5. LLM generates answer
    # 6. Output parser extracts the text answer
    qa_chain = (
        {
            "context": retriever | format_docs,  # Retrieve and format documents
            "question": RunnablePassthrough()     # Pass question through unchanged
        }
        | prompt                                  # Format with prompt template
        | llm                                     # Generate answer with LLM
        | StrOutputParser()                       # Extract text from LLM response
    )
    
    print("[OK] Q&A chain ready!")
    return qa_chain

def main():
    """
    Main function that runs the interactive chat interface.
    """
    
    print("=" * 60)
    print("Constitution of Pakistan - Q&A Agent")
    print("=" * 60)
    print("\nInitializing...")
    
    try:
        # Load the vector store (created by ingest.py)
        vector_store = load_vector_store()
        
        # Create the Q&A chain
        qa_chain = create_qa_chain(vector_store)
        
        print("\n" + "=" * 60)
        print("Ready! You can now ask questions about the Constitution.")
        print("Type 'quit', 'exit', or 'q' to stop.")
        print("=" * 60 + "\n")
        
        # Main interaction loop
        # This runs forever until the user types 'quit'
        while True:
            # Get user input
            question = input("Your question: ").strip()
            
            # Check if user wants to quit
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            # Skip empty questions
            if not question:
                continue
            
            # Process the question
            try:
                # Run the chain with the user's question
                # The chain will:
                # 1. Search for relevant chunks
                # 2. Format them with the question
                # 3. Send to LLM
                # 4. Return the answer
                # With LCEL, we pass the question directly as a string
                answer = qa_chain.invoke(question)
                
                # Remove any "Answer:" prefix if the LLM added it
                # Strip common prefixes that LLMs sometimes add
                answer = answer.strip()
                if answer.lower().startswith("answer:"):
                    answer = answer[7:].strip()  # Remove "Answer:" (7 characters)
                elif answer.lower().startswith("answer "):
                    answer = answer[6:].strip()  # Remove "Answer " (6 characters)
                
                # Display only the answer (no debugging output, no extra formatting)
                print(answer)
                print()  # Empty line for readability
                
            except Exception as e:
                print(f"\n[ERROR] Error processing question: {str(e)}")
                print("Please try again or check your internet connection.\n")
            
            print()  # Empty line for readability
    
    except FileNotFoundError as e:
        print(f"\n[ERROR] Error: {str(e)}")
        print("\nPlease run 'python ingest.py' first to create the vector store.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your OPENAI_API_KEY is set correctly in .env file")
        print("2. Make sure you have an active internet connection")
        print("3. Verify that ingest.py ran successfully")

if __name__ == "__main__":
    # This runs the main function when you execute: python main.py
    main()

