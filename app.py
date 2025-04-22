import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ---- Carregar .env para desenvolvimento local ----
load_dotenv()

# ---- Configuração ----
# Na nuvem Streamlit, defina sua chave OpenAI em Configurações > Segredos como:
#    OPENAI_API_KEY = "sua_chave_api_aqui"
# Localmente, você pode usar um arquivo .env com o mesmo nome de variável.
# ---- Recuperar chave API OpenAI ----
# 1) Tente ler da variável de ambiente (ou arquivo .env)
openai_api_key = os.getenv("OPENAI_API_KEY")

# 2) Se implantado na nuvem Streamlit, tente segredos (dentro de try/except para evitar erros localmente)
try:
    secret_key = st.secrets.get("OPENAI_API_KEY")
    if secret_key:
        openai_api_key = secret_key
except Exception:
    pass

# Validar chave API
if not openai_api_key:
    st.error(
        "🔑 OpenAI API key not found.\n"
        "Set OPENAI_API_KEY in a .env file for local, or in Streamlit Secrets under Settings → Secrets."
    )
    st.stop()

# ---- App Streamlit ----
st.set_page_config(page_title="RAG MBA ENAP - Disciplina IA Generativa", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px 20px; }
    .stTextInput>div>input { border: 2px solid #4CAF50; border-radius: 5px; padding: 10px; }
    .stMarkdown { font-family: 'Arial', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 RAG - Projeto MBA ENAP")
st.subheader("Aluno: José Lobo")

# Sidebar for PDF upload
st.sidebar.header("📄 Upload de Documento")
doc_file = st.sidebar.file_uploader("Envie um documento PDF e converse com ele!", type=["pdf"])

# Resetar histórico de chat e evitar reaplicação da última pergunta ao trocar de arquivo
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0  # Inicializar chave única para o campo de entrada

if doc_file:
    # Verificar se o arquivo enviado é diferente do último carregado
    if st.session_state.last_uploaded_file != doc_file.name:
        st.session_state.chat_history = []  # Limpar histórico de chat
        st.session_state.last_uploaded_file = doc_file.name  # Atualizar o nome do arquivo
        st.session_state.chat_input_key += 1  # Incrementar chave para redefinir o campo de entrada

    # Salvar PDF enviado no disco para PyPDFLoader
    temp_path = "uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(doc_file.getbuffer())

    # Carregar e dividir
    loader = PyPDFLoader(temp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Criar embeddings e armazenamento vetorial FAISS
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Entrada de consulta do usuário com chave única
    question = st.text_input(
        "Ask a question about the document:",
        key=f"chat_input_{st.session_state.chat_input_key}"  # Chave única para redefinir o campo
    )

    if question:
        # Construir cadeia RAG
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0),
            retriever=retriever
        )
        result = qa_chain.invoke({"question": question, "chat_history": st.session_state.chat_history})

        answer = result.get("answer", "No answer returned.")

        # Atualizar histórico de chat
        st.session_state.chat_history.append((question, answer))

    # Exibir conversa
    for q, a in st.session_state.chat_history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
