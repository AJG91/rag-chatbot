SYSTEM_PROMPT = "You are a helpful assistant."

def build_prompt(retrieved_docs, user_question):
    """Builds the output prompt template for the RAG chat."""
    context = "\n".join(retrieved_docs)
    return (
        f"System prompt: {SYSTEM_PROMPT}\n\n"
        f"Use ONLY the information below to answer.\n"
        f"Context:\n{context}\n\n"
        f"User: {user_question}\n"
        f"Assistant:"
    )

def build_prompt_with_history(
    history_turns: list[str], 
    retrieved_docs, 
    user_message: str
)-> str:
    """Builds the output prompt template for the chatbot with previous history."""
    history_text = "\n".join(history_turns)
    context = "\n".join(retrieved_docs)
    if history_text.strip():
        return (
            f"System prompt: {SYSTEM_PROMPT}\n\n"
            f"Use ONLY the information below to answer.\n"
            f"Context:\n{context}\n\n"
            f"{history_text}\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
    else:
        return build_prompt(retrieved_docs, user_message)

def extract_assistant_reply(text: str) -> str:
    """Extracts the chatbot's reply."""
    if "Assistant" in text:
        return text.rsplit("Assistant:", 1)[-1].strip()
    return text.strip()