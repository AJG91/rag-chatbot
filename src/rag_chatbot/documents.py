RAW_DOCUMENTS = [
    "Quantum entanglement describes correlations between particles that cannot be explained classically.",
    "The Higgs boson explains how particles acquire mass through interaction with the Higgs field.",
    "Transformers rely on self-attention mechanisms to model token relationships."
]

def chunk_text(text: str, chunk_size: int = 80) -> list[str]:
    """
    Takes the input string, breaks it up by the `chunk_size`, 
    and creates entries in a list with each string.
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

DOCUMENTS = []
for doc in RAW_DOCUMENTS:
    DOCUMENTS.extend(chunk_text(doc))