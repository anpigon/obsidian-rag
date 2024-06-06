import os
import argparse
from typing import List

from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

import gradio as gr
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get user file path")
    parser.add_argument("--notes_dir", help="Input file path")
    parser.add_argument(
        "--vectorize",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to vectorize the file",
    )
    parser.add_argument("--model", help="Input model name")
    parser.add_argument("--embedding", help="Input embedding model name")
    return parser.parse_args()


def format_docs(docs: List[dict]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def remove_all_files_in_folder(directory: str) -> None:
    os.system(f"rm -rf {directory}/*")


def load_vectorstore(
    docs_path: str, embedding_model_name: str, vectorize: bool
) -> Chroma:
    print("vectorize:", vectorize)
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )
    store = LocalFileStore("./.cached_embeddings/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model_name
    )

    if vectorize:
        loader = ObsidianLoader(path=docs_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200
        )
        all_splits = text_splitter.split_documents(data)

        # Hard reset cause LLM be weird
        remove_all_files_in_folder("vectorstore")

        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=cached_embeddings,
            persist_directory="vectorstore",
        )
        print("Vectorized!")
    else:
        vectorstore = Chroma(
            embedding_function=cached_embeddings,
            persist_directory="vectorstore",
        )
        print("Loaded vectorstore!")

    return vectorstore


def main(
    question: str,
    notes_dir: str = None,
    vectorize: bool = False,
    model: str = None,
    embedding: str = None,
) -> str:
    docs_path = notes_dir
    model_name = model if model else "aya"
    embedding_model_name = embedding if embedding else "BAAI/bge-m3"

    vectorstore = load_vectorstore(docs_path, embedding_model_name, vectorize)

    rag_prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)


# Gradio interface
demo = gr.Interface(
    fn=lambda question: main(
        question, args.notes_dir, args.vectorize, args.model, args.embedding
    ),
    inputs="text",
    outputs="text",
)

if __name__ == "__main__":
    args = get_args()
    demo.launch(show_api=False)
