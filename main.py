import json
from urllib.parse import quote

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

import os
from operator import itemgetter
from pathlib import Path

import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_teddynote import logging
from langchain_teddynote.retrievers import KiwiBM25Retriever

from document_loaders.obsidian import MyObsidianLoader

logging.langsmith("obsidian-rag", set_enable=True)

root_path = Path.cwd()

embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
answer_model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

# 설정 파일 경로
CONFIG_FILE = Path.home() / ".obsidian_rag_config.json"


# Chat history
msgs = StreamlitChatMessageHistory(key="chat_messages")


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"last_path": "", "saved_paths": []}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


# Streamlit app setup
st.set_page_config(page_title="Obsidian RAG Chatbot", page_icon=":books:")
st.title("Obsidian RAG Chatbot")

# 설정 로드
config = load_config()

# Initialize embedding model
# model_name = "BAAI/bge-m3"
# model_kwargs = {"device": "mps"}
# encode_kwargs = {"normalize_embeddings": True}
# underlying_embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs,
# )
underlying_embeddings = OpenAIEmbeddings(model=embedding_model_name)

# Initialize cache storage
store = LocalFileStore(root_path / ".cached_embeddings")

# Create cached embeddings
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=store,
    namespace=(
        underlying_embeddings.model_name
        if "model_name" in underlying_embeddings
        else underlying_embeddings.model
    ),
)

# Initialize OpenAI model
llm = ChatOpenAI(model_name=answer_model_name, temperature=0.1)


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
        # persist_directory="./.vectorstore",
    )

    bm25_retriever = KiwiBM25Retriever.from_documents(texts)

    for i, _ in enumerate(texts):
        progress = (i + 1) / total_texts
        progress_bar.progress(progress)

    st.success("Embedding completed!")
    return (vectorstore, bm25_retriever)


# Sidebar for Obsidian folder path input
import streamlit as st

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
