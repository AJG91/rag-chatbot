def build_rag_prompt(system_prompt, retrieved_docs, user_question):
    context = "\n".join(retrieved_docs)
    return (
        f"{system_prompt}\n\n"
        f"Use ONLY the information below to answer.\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}\n"
        f"Answer:"
    )