import json
from pathlib import Path

def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def iter_jsonl_directory(directory: str) -> list[str]:
    directory = Path(directory)
    
    all_text = []
    for file_path in directory.glob("*.jsonl"):
        all_text.append(preprocess_documents(file_path))
    
    return [text for subtext in all_text for text in subtext]

def chunk_text(text: str, chunk_size: int = 80) -> list[str]:
    """
    Takes the input string, breaks it up by the `chunk_size`, 
    and creates entries in a list with each string.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def preprocess_documents(path: str) -> list[str]:
    DOCUMENTS = []
    for doc in iter_jsonl(path):
        DOCUMENTS.extend(chunk_text(doc["text"]))
    return DOCUMENTS