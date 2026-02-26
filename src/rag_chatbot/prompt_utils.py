SYSTEM_PROMPT = "You are a helpful assistant."

def build_prompt(retrieved_docs, user_question):
    """Builds the output prompt template for the RAG chat."""
    context = "\n".join(retrieved_docs)
    return (
        f"System prompt: {SYSTEM_PROMPT}\n\n"
        f"Use ONLY the information below to answer.\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n"
        f"Answer:"
    )