import json
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

import os
from operator import itemgetter
from pathlib import Path

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_teddynote import logging
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_upstage.embeddings import UpstageEmbeddings
from streamlit_extras.stylable_container import stylable_container

from document_loaders.obsidian import MyObsidianLoader

logging.langsmith("obsidian-rag", set_enable=True)

root_path = Path.cwd()

answer_model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

# 설정 파일 경로
CONFIG_FILE = Path.home() / ".obsidian_rag_config.json"

# Embedding model options
EMBEDDING_MODELS = {
    "OpenAI text-embedding-3-small": ("openai", "text-embedding-3-small"),
    "OpenAI text-embedding-3-large": ("openai", "text-embedding-3-large"),
    "HuggingFace BAAI/bge-m3": ("hf_bge", "BAAI/bge-m3"),
    "HuggingFace intfloat/multilingual-e5-large-instruct": (
        "hf",
        "intfloat/multilingual-e5-large-instruct",
    ),
    "HuggingFace intfloat/multilingual-e5-large": (
        "hf",
        "intfloat/multilingual-e5-large",
    ),
    "Upstage solar-embedding-1-large": ("upstage", "solar-embedding-1-large"),
    "Cohere embed-multilingual-v3.0": ("cohere", "embed-multilingual-v3.0"),
}

# Chat history
msgs = StreamlitChatMessageHistory(key="chat_messages")


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Set default values if keys are missing
    if "last_path" not in config:
        config["last_path"] = ""
    if "saved_paths" not in config:
        config["saved_paths"] = []
    if (
        "last_embedding_model" not in config
        or config["last_embedding_model"] not in EMBEDDING_MODELS
    ):
        config["last_embedding_model"] = list(EMBEDDING_MODELS.keys())[
            0
        ]  # Default to the first model

    return config


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


# Streamlit app setup
st.set_page_config(page_title="Obsidian RAG Chatbot", page_icon=":books:")
st.title("Obsidian RAG Chatbot")

# 설정 로드
config = load_config()

# Initialize cache storage
store = LocalFileStore(root_path / ".cached_embeddings")


# Initialize embedding model based on selection
def initialize_embeddings(model_type, model_name):
    model_kwargs = {"device": "mps"}  # Change to "cuda" if using GPU
    encode_kwargs = {"normalize_embeddings": False}

    if model_type == "openai":
        return OpenAIEmbeddings(model=model_name)
    elif model_type == "hf_bge":
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif model_type == "hf":
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif model_type == "upstage":
        return UpstageEmbeddings(model_name=model_name)
    elif model_type == "cohere":
        return CohereEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Load Obsidian notes and create vector store
def load_vectorstore(obsidian_path: str) -> tuple[VectorStore, KiwiBM25Retriever]:
    loader = MyObsidianLoader(obsidian_path, encoding="utf-8", collect_metadata=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=1024, chunk_overlap=24, language=Language.MARKDOWN
    )
    texts = text_splitter.split_documents(documents)

    progress_bar = st.progress(0)
    total_texts = len(texts)

    vectorstore = Chroma.from_documents(
        texts,
        cached_embeddings,
    )

    bm25_retriever = KiwiBM25Retriever.from_documents(texts)

    for i, _ in enumerate(texts):
        progress = (i + 1) / total_texts
        progress_bar.progress(progress)

    st.success("Embedding completed!")
    return (vectorstore, bm25_retriever)


# Sidebar for Obsidian folder path input
with st.sidebar:
    st.header("📚 Obsidian Vault Settings")

    # Container for path selection and input
    with st.container():
        # Saved paths dropdown
        saved_paths = config["saved_paths"]
        selected_path = st.selectbox(
            "📂 Choose a saved Obsidian path:",
            options=[""] + saved_paths,
            index=(
                0
                if config["last_path"] not in saved_paths
                else saved_paths.index(config["last_path"]) + 1
            ),
            key="saved_path_select",
        )

        # New path input
        new_path = st.text_input(
            "🆕 Or enter a new Obsidian folder path:",
            config["last_path"],
            key="new_path_input",
        )

        # Path actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "💾 Save Path", key="save_path_button", use_container_width=True
            ):
                if new_path and new_path not in saved_paths:
                    saved_paths.append(new_path)
                    config["saved_paths"] = saved_paths
                    config["last_path"] = new_path
                    save_config(config)
                    st.success(f"Path '{new_path}' saved!")
                    st.rerun()

        with col2:
            if st.button(
                "🗑️ Clear Paths", key="clear_paths_button", use_container_width=True
            ):
                config["saved_paths"] = []
                config["last_path"] = ""
                save_config(config)
                st.success("All saved paths cleared!")
                st.rerun()

    # Selected path display
    obsidian_path = selected_path or new_path
    if obsidian_path:
        st.info(f"📁 Current path: {obsidian_path}")

    # Embedding model selection
    st.subheader("Embedding Model Selection")
    selected_model = st.selectbox(
        "Choose an embedding model:",
        options=list(EMBEDDING_MODELS.keys()),
        index=list(EMBEDDING_MODELS.keys()).index(config["last_embedding_model"]),
        key="embedding_model_select",
    )

    embedding_model_type, embedding_model_name = EMBEDDING_MODELS[selected_model]

    # Save selected model to config
    if selected_model != config["last_embedding_model"]:
        config["last_embedding_model"] = selected_model
        save_config(config)

    # Initialize the selected embedding model
    underlying_embeddings = initialize_embeddings(
        embedding_model_type, embedding_model_name
    )

    # Create cached embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=store,
        namespace=getattr(
            underlying_embeddings,
            "model",
            getattr(underlying_embeddings, "model_name", embedding_model_name),
        ),
    )

    # Embedding button
    if st.button(
        "🚀 Start Embedding", key="start_embedding_button", use_container_width=True
    ):
        with st.spinner("Embedding in progress..."):
            (vectorstore, bm25_retriever) = load_vectorstore(obsidian_path)
        st.session_state.vectorstore = vectorstore
        st.session_state.bm25_retriever = bm25_retriever

    # Reset conversation button
    if st.button(
        "🔄 Reset Conversation",
        key="reset_conversation_button",
        use_container_width=True,
    ):
        msgs.clear()
        st.success("Conversation reset!")
        st.rerun()

# Initialize OpenAI model
llm = ChatOpenAI(model_name=answer_model_name, temperature=0.1)


def create_obsidian_link(file_path: str, obsidian_vault_path: str) -> str:
    relative_path = os.path.relpath(file_path, obsidian_vault_path)
    encoded_path = quote(relative_path)
    return f"obsidian://open?vault={os.path.basename(obsidian_vault_path)}&file={encoded_path}"


# RAG chain setup
def rag_chain():
    prompt_template = """You are an assistant whose primary purpose is to help with questions or inquiries about notes written in Markdown. Base your answer on the provided CONTEXT and the chat history.

    Use the following context snippets to answer the question:
    <CONTEXT>
    {context}
    </CONTEXT>

    After your response, provide the sources of your information in the following format:
    **Sources:**
    - [note title 1](obsidian://open?path=url-encoded_file_absolute_path)
    - [note title 2](obsidian://open?path=url-encoded_file_absolute_path)
    ...

    Ensure each source is on a new line and follows the Markdown link format.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # VectorStoreRetriever
    vectorstore_retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 4}
    )

    # BM25Retriever
    bm25_retriever = st.session_state.bm25_retriever

    # EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=[0.4, 0.6],
        search_type="mmr",
    )

    # MultiQueryRetriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | multi_query_retriever
        )
        | {
            "context": itemgetter("context"),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="chat_history",
    )


# Handle user input
def handle_user_input(user_question):
    st.chat_message("human").write(user_question)

    if "vectorstore" in st.session_state:
        chain = rag_chain()
        config = {"configurable": {"session_id": "default"}}
        response = chain.stream({"question": user_question}, config=config)
        with st.chat_message("ai"):
            container = st.empty()
            answer = ""
            for message in response:
                answer += message
                container.markdown(answer)
    else:
        st.warning("Please embed your Obsidian folder first.")


# Main interface
st.header("💬 Chat with your Obsidian Notes")

# Display previous messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

user_question = st.chat_input("Ask a question about your notes.")
if user_question:
    handle_user_input(user_question)
