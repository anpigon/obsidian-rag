import argparse
import os
from typing import List

import gradio as gr
from dotenv import load_dotenv
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.storage import LocalFileStore
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

load_dotenv()


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
    documents: List[Document], embedding_model_name: str, vectorize: bool
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
        # Hard reset cause LLM be weird
        remove_all_files_in_folder("vectorstore")

        vectorstore = Chroma.from_documents(
            documents=documents,
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

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=2000, chunk_overlap=200, language=Language.MARKDOWN
    )
    docs = ObsidianLoader(path=docs_path).load_and_split(text_splitter)

    vectorstore = load_vectorstore(docs, embedding_model_name, vectorize)

    # EnsembleRetriever
    vectorstore_retriever = vectorstore.as_retriever()
    bm25_retriever = BM25Retriever.from_documents(docs)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vectorstore_retriever],
        weights=[0.4, 0.6],
        search_type="mmr",
    )

    llm = ChatOllama(model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    # MultiQueryRetriever
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=ensemble_retriever,
        llm=llm,
    )

    rag_prompt = hub.pull("rlm/rag-prompt")
    qa_chain = (
        {
            "context": multi_query_retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)


# Gradio interface
demo = gr.Interface(
    fn=lambda question: main(
        question=question,
        notes_dir=args.notes_dir,
        vectorize=args.vectorize,
        model=args.model,
        embedding=args.embedding,
    ),
    inputs="text",
    outputs="text",
)

if __name__ == "__main__":
    args = get_args()
    demo.launch(show_api=False)


# if __name__ == "__main__":
#     args = get_args()
#     while True:
#         question = input("Enter your question (or type 'exit' to quit): ")
#         if question.lower() == "exit":
#             break
#         answer = main(
#             question=question,
#             notes_dir=args.notes_dir,
#             vectorize=args.vectorize,
#             model=args.model,
#             embedding=args.embedding,
#         )
#         print("Answer:", answer)
