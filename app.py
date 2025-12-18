"""
Streamlit Web Application for Constitution of Pakistan Q&A Agent

This is a web-based chat interface for asking questions about the Constitution of Pakistan.
It uses the same RAG backend as main.py but provides a modern web UI using Streamlit.

Run this with: streamlit run app.py
"""

# Suppress Pydantic warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core.*")

import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import config

# Page configuration
st.set_page_config(
    page_title="Pakistan Constitution Q&A",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to fix input field styling (remove red border on focus)
st.markdown("""
    <style>
        /* Comprehensive CSS to remove red border from Streamlit chat input */
        /* Target all possible input selectors */
        .stChatInput input:focus,
        .stChatInput textarea:focus,
        .stChatInput > div > div > input:focus,
        .stChatInput > div > div > textarea:focus,
        .stChatInput input:focus-visible,
        .stChatInput textarea:focus-visible,
        div[data-baseweb="input"] input:focus,
        div[data-baseweb="input"] textarea:focus,
        div[data-baseweb="input"] input:focus-visible,
        div[data-baseweb="input"] textarea:focus-visible,
        input[data-testid="stChatInputTextInput"]:focus,
        textarea[data-testid="stChatInputTextInput"]:focus {
            border-color: rgb(38, 39, 48) !important;
            box-shadow: 0 0 0 1px rgb(38, 39, 48) !important;
            outline: none !important;
        }
        
        /* Remove red border in all states */
        .stChatInput input,
        .stChatInput textarea,
        .stChatInput > div > div > input,
        .stChatInput > div > div > textarea,
        div[data-baseweb="input"] input,
        div[data-baseweb="input"] textarea,
        input[data-testid="stChatInputTextInput"],
        textarea[data-testid="stChatInputTextInput"] {
            border-color: rgb(38, 39, 48) !important;
        }
        
        /* Remove invalid/error styling */
        .stChatInput input:invalid,
        .stChatInput textarea:invalid,
        .stChatInput > div > div > input:invalid,
        .stChatInput > div > div > textarea:invalid,
        div[data-baseweb="input"] input:invalid,
        div[data-baseweb="input"] textarea:invalid {
            border-color: rgb(38, 39, 48) !important;
            box-shadow: none !important;
        }
        
        /* Target the baseweb input wrapper */
        div[data-baseweb="input"],
        div[data-baseweb="input"]:focus-within,
        div[data-baseweb="input"]:focus {
            border-color: rgb(38, 39, 48) !important;
            box-shadow: 0 0 0 1px rgb(38, 39, 48) !important;
        }
        
        /* Override any red color specifically */
        .stChatInput * {
            --border-color: rgb(38, 39, 48) !important;
        }
        
        /* Remove any red border color */
        *[style*="border-color: red"],
        *[style*="border-color: #ff0000"],
        *[style*="border-color: rgb(255, 0, 0)"] {
            border-color: rgb(38, 39, 48) !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vector_store_loaded" not in st.session_state:
    st.session_state.vector_store_loaded = False


@st.cache_resource
def load_vector_store():
    """
    Load the FAISS vector store from disk.
    This function is cached so it only runs once per session.
    """
    # Check if vector store exists
    if not os.path.exists(config.VECTOR_DB_PATH):
        return None
    
    # Create embeddings model (same as used during ingestion)
    embeddings = OpenAIEmbeddings(
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Load the FAISS vector store from disk
    vector_store = FAISS.load_local(
        folder_path=config.VECTOR_DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vector_store


def create_qa_chain(vector_store):
    """
    Create a RetrievalQA chain that combines retrieval and question answering.
    Uses the same logic as main.py.
    """
    # Create the LLM
    llm = ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=config.TEMPERATURE,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # System prompt - same as finalized in main.py
    system_prompt = """You are a Pakistan Constitution expert. Answer the user's question directly and briefly. Do not add prefixes like "Answer:" or "Response:" - just provide the answer directly. Do not provide extra background unless asked. If the user asks 'What is Article 6', just state the law of High Treason simply. Do not mention that you are an AI. If a footnote or later text says an article was 'omitted' or 'substituted', prioritize the amendment over the original text. If the answer is not in the document, state that you don't know."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\nContext from Constitution:\n{context}"),
        ("human", "{question}")
    ])
    
    # Create the Retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks
    )
    
    # Format documents function
    def format_docs(docs):
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the Q&A Chain using LCEL
    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain


def clean_answer(answer):
    """
    Remove any "Answer:" prefix if the LLM added it.
    """
    answer = answer.strip()
    if answer.lower().startswith("answer:"):
        answer = answer[7:].strip()
    elif answer.lower().startswith("answer "):
        answer = answer[6:].strip()
    return answer


# Sidebar
with st.sidebar:
    st.title("⚙️ Options")
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # About section
    st.subheader("📖 About")
    st.markdown("""
    This is an AI-powered Q&A system for the **Constitution of Pakistan**.
    
    Ask questions about any Article, Amendment, or legal provision, and get
    accurate answers based on the official Constitution document.
    
    **Features:**
    - ✅ Direct answers from the Constitution
    - ✅ Amendment-aware responses
    - ✅ ChatGPT-like interface
    """)
    
    st.divider()
    
    # Status indicator
    if st.session_state.vector_store_loaded:
        st.success("✅ Vector store loaded")
    else:
        st.warning("⚠️ Loading vector store...")


# Main content area
st.title("📜 Pakistan Constitution Q&A")
st.markdown("Ask questions about the Constitution of Pakistan")

# Additional JavaScript to force remove red border (runs after page load)
st.markdown("""
    <script>
        // Function to remove red borders from input fields
        function removeRedBorders() {
            const inputs = document.querySelectorAll('.stChatInput input, .stChatInput textarea, div[data-baseweb="input"] input, div[data-baseweb="input"] textarea');
            inputs.forEach(input => {
                input.addEventListener('focus', function() {
                    this.style.borderColor = 'rgb(38, 39, 48)';
                    this.style.boxShadow = '0 0 0 1px rgb(38, 39, 48)';
                    this.style.outline = 'none';
                });
                input.addEventListener('blur', function() {
                    this.style.borderColor = 'rgb(38, 39, 48)';
                });
            });
        }
        
        // Run on page load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', removeRedBorders);
        } else {
            removeRedBorders();
        }
        
        // Also run after Streamlit renders (using MutationObserver)
        const observer = new MutationObserver(removeRedBorders);
        observer.observe(document.body, { childList: true, subtree: true });
    </script>
""", unsafe_allow_html=True)

# Load vector store and create chain (only once)
if not st.session_state.vector_store_loaded:
    with st.spinner("Loading vector store..."):
        try:
            vector_store = load_vector_store()
            
            if vector_store is None:
                st.error(f"""
                ❌ **Vector store not found!**
                
                The FAISS index was not found at: `{config.VECTOR_DB_PATH}`
                
                Please run `python ingest.py` first to create the vector store.
                """)
                st.stop()
            
            # Create Q&A chain
            st.session_state.qa_chain = create_qa_chain(vector_store)
            st.session_state.vector_store_loaded = True
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading vector store: {str(e)}")
            st.info("""
            **Troubleshooting:**
            1. Make sure you've run `python ingest.py` first
            2. Check that your `.env` file has the correct `OPENAI_API_KEY`
            3. Verify the vector store folder exists
            """)
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the Constitution..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get answer from the chain
                answer = st.session_state.qa_chain.invoke(prompt)
                
                # Clean the answer (remove "Answer:" prefix if present)
                answer = clean_answer(answer)
                
                # Display the answer
                st.markdown(answer)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

