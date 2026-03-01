from pathlib import Path
from utils import iter_jsonl

def iter_jsonl_directory(directory: str) -> list[str]:
    """
    Reads all jsonl files in a directory and processes them
    to be used for retrieval-augmented generation (RAG).

    Parameters
    ----------
    path : str
        Path to directory where jsonl files are contained.

    Returns
    -------
    list[str]
        Post-processed text read from jsonl files.
    """
    directory = Path(directory)
    
    all_text = []
    for file_path in directory.glob("*.jsonl"):
        all_text.append(process_documents(file_path))
    
    return [text for subtext in all_text for text in subtext]

def process_documents(path: str) -> list[str]:
    """
    Iterates through the different jsonl files and breaks the text into chunks.

    Parameters
    ----------
    path : str
        Path to directory where jsonl files are contained.

    Returns
    -------
    list[str]
        Post-processed text.
    """
    DOCUMENTS = []
    for doc in iter_jsonl(path):
        DOCUMENTS.extend(chunk_text(doc["text"]))
    return DOCUMENTS

def chunk_text(text: str, chunk_size: int = 80) -> list[str]:
    """
    Takes the input string, breaks it up by the `chunk_size`, 
    and creates entries in a list with each string.

    Parameters
    ----------
    text : str
    chunk_size : int

    Returns
    -------
    list[str]
        Text that has been split into chunks.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]