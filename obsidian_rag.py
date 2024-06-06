import os
import glob
import argparse

from typing import Dict, List, Tuple, Union

from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_core.callbacks import StreamingStdOutCallbackHandler

import gradio as gr


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Get user file path"
    )
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


def format_docs(docs: List[str]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def remove_all_files_in_folder(directory: str) -> None:
    os.system(f"rm -rf {directory}/*")


def main(question: str) -> str:
    args = get_args()
    docs_path = args.notes_dir
    model_name = args.model if args.model != None else "aya"
    embedding_model_name = args.embedding if args.embedding != None else "BAAI/bge-m3"

    embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if args.vectorize:
        loader: ObsidianLoader = ObsidianLoader(path=docs_path)
        data: List[str] = loader.load()
        text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200
        )
        all_splits: List[str] = text_splitter.split_documents(data)

        # Hard reset cause LLM be weird
        remove_all_files_in_folder("vectorstore")

        vectorstore: Chroma = Chroma.from_documents(
            documents=all_splits,
            embedding_function=embedding,
            persist_directory="vectorstore",
        )
        print("Vectorized!")

    else:
        vectorstore: Chroma = Chroma(
            embedding_function=embedding,
            persist_directory="vectorstore",
        )
        print("Loaded vectorstore!")

    rag_prompt: str = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model=model_name, callbacks=[StreamingStdOutCallbackHandler()])

    retriever = vectorstore.as_retriever()
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(question)


# TODO: seperate out loading function for local vectorstore
demo = gr.Interface(fn=main, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch(show_api=False)
