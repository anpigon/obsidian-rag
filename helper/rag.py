import math
import re

import tqdm
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from document_loaders.obsidian import MyObsidianLoader
from retrievers.kiwi_bm25 import KiwiBM25Retriever

PERSIST_DIRECTORY = "./.vectorstore/"


def create_embeddings(model_name="BAAI/bge-m3", cache_path="./.cache"):
    underlying_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": False},
    )

    embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        batch_size=100,
        document_embedding_cache=LocalFileStore(cache_path),
        namespace=underlying_embeddings.model_name,
    )

    return embeddings


def load_documents(path, encoding="utf-8", collect_metadata=True):
    """주어진 경로에서 Obsidian 문서를 로드합니다.

    이 함수는 MyObsidianLoader를 사용하여 지정된 경로에서 Obsidian 문서를 로드합니다.
    문서 인코딩과 메타데이터 수집 여부를 선택적으로 지정할 수 있습니다.

    Args:
        path (str): Obsidian 문서가 저장된 디렉토리 경로
        encoding (str, optional): 문서 인코딩 방식. 기본값은 "utf-8"입니다.
        collect_metadata (bool, optional): 메타데이터 수집 여부. 기본값은 True입니다.

    Returns:
        list: 로드된 문서 객체의 리스트

    Example:
        documents = load_documents("/path/to/obsidian/vault")
    """
    loader = MyObsidianLoader(
        path=path, encoding=encoding, collect_metadata=collect_metadata
    )
    return loader.load()


def split_documents(docs, embeddings):
    """
    주어진 문서를 의미론적 청크로 분할합니다.

    이 함수는 SemanticChunker를 사용하여 입력 문서를 더 작은 의미 있는 단위로 나눕니다.
    각 청크는 원본 문서의 의미를 유지하면서 독립적으로 처리될 수 있습니다.

    Args:
        docs (List[Document]): 분할할 문서 리스트
        embeddings (Embeddings): 텍스트 임베딩을 생성하는 데 사용할 임베딩 모델

    Returns:
        List[Document]: 분할된 문서 청크의 리스트

    Example:
        embeddings = HuggingFaceEmbeddings()
        documents = [Document(page_content="긴 텍스트 내용...")]
        chunks = split_documents(documents, embeddings)
    """
    text_splitter = SemanticChunker(embeddings=embeddings)
    return text_splitter.split_documents(docs)


def create_vector_store(
    documents,
    embeddings,
    collection_name,
    batch_size=10000,
    persist_directory=PERSIST_DIRECTORY,
) -> VectorStore:
    """
    문서를 벡터 저장소에 임베딩하고 인덱싱하는 함수입니다.

    이 함수는 주어진 문서들을 배치 단위로 처리하여 Chroma 벡터 저장소에 추가합니다.
    진행 상황은 tqdm 프로그레스 바를 통해 표시됩니다.

    Args:
        documents (list): 인덱싱할 문서 리스트
        embeddings (Embeddings): 문서를 임베딩하는 데 사용할 임베딩 함수
        collection_name (str): Chroma 컬렉션의 이름
        batch_size (int, optional): 각 배치에서 처리할 문서의 수. 기본값은 10000입니다.

    Returns:
        VectorStore: 생성된 Chroma 벡터 저장소 객체

    Note:
        이 함수는 대량의 문서를 효율적으로 처리하기 위해 배치 처리 방식을 사용합니다.
    """
    vector_store = Chroma(
        collection_name=sanitize_collection_name(collection_name),
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(documents)
    # total_size = len(documents)
    # num_batches = math.ceil(total_size / batch_size)
    # with tqdm(total=num_batches, desc="Indexing") as pbar:
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = min((i + 1) * batch_size, total_size)
    #         batch_documents = documents[start_idx:end_idx]
    #         vector_store.add_documents(batch_documents)
    #         pbar.update(1)
    return vector_store


def sanitize_collection_name(path: str) -> str:
    # 1. 특수 문자를 언더스코어로 대체
    name = re.sub(r"[^\w\-]", "_", path)

    # 2. 연속된 언더스코어를 하나로 축소
    name = re.sub(r"_+", "_", name)

    # 3. 시작과 끝의 언더스코어 제거
    name = name.strip("_")

    # 4. 소문자로 변환
    name = name.lower()

    # 5. 이름이 숫자로 시작하면 앞에 'c_' 추가
    if name[0].isdigit():
        name = "c_" + name

    # 6. 이름이 비어있거나 너무 짧으면 기본값 사용
    if len(name) < 3:
        name = "default_collection"

    # 7. 이름이 63자를 초과하면 자르기
    name = name[:63]

    return name


# %%
def create_retriever(path: str, embeddings: Embeddings) -> BaseRetriever:
    # 문서 불러오기
    docs = load_documents(path)

    # 문서 분할
    chunks = split_documents(docs, embeddings)

    # 벡터 스토어 생성
    vector_store = create_vector_store(
        chunks,
        embeddings,
        collection_name=path,
        persist_directory=PERSIST_DIRECTORY,
    )

    # Retriver 생성
    dense_retriver = vector_store.as_retriever(k=150)
    sparse_retriever = KiwiBM25Retriever.from_documents(documents=chunks, k=150)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriver, sparse_retriever],
        weights=[0.4, 0.6],
    )

    # 리랭커
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=20)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )
    return compression_retriever
