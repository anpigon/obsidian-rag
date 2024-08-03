import os
from operator import itemgetter
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStore
from langchain_teddynote import logging
from langchain_teddynote.retrievers import KiwiBM25Retriever

from document_loaders.obsidian import MyObsidianLoader
from helper.constants import EMBEDDING_MODELS
from helper.initialize_embeddings import initialize_embeddings
from helper.load_config import load_config, save_config
from helper.utils import format_documents

load_dotenv()

logging.langsmith("obsidian-rag", set_enable=True)

root_path = Path.cwd()


# 설정 로드
config = load_config()

# Initialize cache storage
store = LocalFileStore(root_path / ".cached_embeddings")

answer_model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

# Chat history
msgs = StreamlitChatMessageHistory(key="chat_messages")

# Streamlit app setup
st.set_page_config(page_title="Obsidian RAG Chatbot", page_icon=":books:")
st.title("Obsidian RAG Chatbot")

# Main interface
st.header("💬 Chat with your Obsidian Notes")


# Load Obsidian notes and create vector store
@st.cache_resource(show_spinner="Loading Obsidian notes...")
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
        embeddings,
    )
    print("Vectorstore created!")

    # 진행률 업데이트
    progress_bar.progress(0.5)  # Vectorstore 생성 후 50%로 업데이트

    bm25_retriever = KiwiBM25Retriever.from_documents(texts)
    bm25_retriever.k = 10
    print("BM25 retriever created!")

    # 진행률 업데이트
    progress_bar.progress(1.0)  # BM25 retriever 생성 후 100%로 업데이트

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
    embeddings = initialize_embeddings(
        embedding_model_type,
        embedding_model_name,
    )

    # Embedding button
    if st.button(
        "🚀 Start Embedding", key="start_embedding_button", use_container_width=True
    ):
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


# RAG chain setup
def generate_rag_chain(retriever: BaseRetriever) -> RunnableWithMessageHistory:
    prompt_template = load_prompt("prompts/obsidian_rag.yaml")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # MultiQueryRetriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
    )

    rag_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | multi_query_retriever | format_documents
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


def generate_retriever(
    vectorstore: VectorStore, bm25_retriever: BM25Retriever
) -> BaseRetriever:
    # VectorStoreRetriever
    vectorstore_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 10}
    )

    # EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=[0.6, 0.4],
        search_type="mmr",
    )

    # TODO: add reranker

    return ensemble_retriever


# Handle user input
def handle_user_input(user_question):
    st.chat_message("human").write(user_question)

    if "vectorstore" in st.session_state and "bm25_retriever" in st.session_state:
        retriever = generate_retriever(
            st.session_state.vectorstore, st.session_state.bm25_retriever
        )

        rag_chain = generate_rag_chain(retriever)

        config = {"configurable": {"session_id": "default"}}
        response = rag_chain.stream({"question": user_question}, config=config)

        with st.chat_message("ai"):
            container = st.empty()
            answer = ""
            for message in response:
                answer += message
                container.markdown(answer)
    else:
        st.warning("Please embed your Obsidian folder first.")


# Display previous messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

user_question = st.chat_input("Ask a question about your notes.")
if user_question:
    handle_user_input(user_question)
