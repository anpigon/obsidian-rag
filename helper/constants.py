from pathlib import Path

# 설정 파일 경로
CONFIG_FILE_PATH = Path.home() / ".obsidian_rag_config.json"

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
