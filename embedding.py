# %%
from dotenv import load_dotenv

load_dotenv()

import argparse
from pathlib import Path

import streamlit as st
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore


from langchain_text_splitters import Language

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from document_loaders.obsidian import MyObsidianLoader

# %%
root_path = Path.cwd()

# %%
store = LocalFileStore(root_path / ".cached_embeddings")

# %%
model_name = "intfloat/multilingual-e5-large-instruct"
model_kwargs = {"device": "mps"}  # Change to "cuda" if using GPU
encode_kwargs = {"normalize_embeddings": False}

underlying_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# %%
namespace = getattr(
    underlying_embeddings,
    "model",
    getattr(underlying_embeddings, "model_name", model_name),
)


# %%
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=store,
    namespace=getattr(
        underlying_embeddings,
        "model",
        getattr(underlying_embeddings, "model_name", model_name),
    ),
)


def main(obsidian_path: str):
    loader = MyObsidianLoader(obsidian_path, encoding="utf-8", collect_metadata=True)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=1024, chunk_overlap=24, language=Language.MARKDOWN
    )
    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        texts,
        cached_embeddings,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Obsidian Notes.")
    parser.add_argument("obsidian_path", type=str, help="Path to the Obsidian note")
    args = parser.parse_args()

    main(args.obsidian_path)
