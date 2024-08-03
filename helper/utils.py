from urllib.parse import quote
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_obsidian_link(file_path: str) -> str:
    encoded_path = quote(file_path)
    return f"obsidian://open?path={encoded_path}"


def format_documents(docs):
    formatted_docs = []
    for doc in docs:
        formatted_doc = f"**Title:** {doc.metadata['source']}\n**Content:**\n {doc.page_content}\n**Source:** [{doc.metadata['source']}]({create_obsidian_link(doc.metadata['path'])})"
        formatted_docs.append(formatted_doc)
    return ("\n" + "-" * 50 + "\n").join(formatted_docs)
