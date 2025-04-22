import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ---- Load .env for local development ----
load_dotenv()

# ---- Configuration ----
# On Streamlit Cloud, set your OpenAI key under Settings > Secrets as:
#    OPENAI_API_KEY = "your_api_key_here"
# Locally, you can use a .env file with the same variable name.
# ---- Retrieve OpenAI API key ----
# 1) Try reading from environment variable (or .env file)
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2) If deployed on Streamlit Cloud, try secrets (inside try/except to avoid errors locally)
try:
    secret_key = st.secrets.get("OPENAI_API_KEY")
    if secret_key:
        openai_api_key = secret_key
except Exception:
    pass

# Validate API key
if not openai_api_key:
    st.error(
        "🔑 OpenAI API key not found.\n"
        "Set OPENAI_API_KEY in a .env file for local, or in Streamlit Secrets under Settings → Secrets."
    )
    st.stop()

# ---- Streamlit App ----
st.set_page_config(page_title="Simple RAG", layout="wide")
st.title("📚 Simple RAG: Streamlit + LangChain + OpenAI")

# Upload PDF
doc_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if doc_file:
    # Save uploaded PDF to disk for PyPDFLoader
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(doc_file.getbuffer())

    # Load and split
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Initialize or retrieve chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User query input
    question = st.text_input("Ask a question about the document:")
    if question:
        # Build RAG chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
            retriever=retriever
        )
        result = qa_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})

        answer = result.get("answer", "No answer returned.")

        # Update chat history
        st.session_state.chat_history.append((question, answer))

    # Display conversation
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
