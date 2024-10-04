# Initialize embedding model based on selection
import os
from pathlib import Path

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_cohere import CohereEmbeddings
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings

from helper.utils import get_device

root_path = Path.cwd()

store_path = root_path / ".cached/embeddings"

if not store_path.exists():
    os.makedirs(store_path)

store = LocalFileStore(store_path)


def initialize_embeddings(model_type, model_name):
    model_kwargs = {"device": get_device()}  # Change to "cuda" if using GPU
    encode_kwargs = {"normalize_embeddings": True}

    underlying_embeddings: BaseModel = None
    if model_type == "openai":
        underlying_embeddings = OpenAIEmbeddings(model=model_name)
    elif model_type == "hf_bge":
        underlying_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif model_type == "hf":
        underlying_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    elif model_type == "upstage":
        underlying_embeddings = UpstageEmbeddings(model_name=model_name)
    elif model_type == "cohere":
        underlying_embeddings = CohereEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create cached embeddings
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=store,
        namespace=getattr(
            underlying_embeddings,
            "model",
            getattr(underlying_embeddings, "model_name", model_name),
        ),
    )

    return cached_embeddings
