from basic_chatbot.model_local import LocalLM
from basic_chatbot.model_openai import OpenAIChat
from rag_chatbot.prompt_utils import build_prompt

class RAGChat():
    def __init__(
        self, 
        model_name: str, 
        retriever,
        api_key: str | None = None
    ):
        if api_key:
            self.lm = OpenAIChat(model_name, api_key=api_key)
        else:
            self.lm = LocalLM(model_name)

        self.retriever = retriever

    def ask(self, question: str) -> str:
        docs = self.retriever.retrieve(question)
        prompt = build_prompt(docs, question)
        output = self.lm.generate_text(prompt)
        reply = self.extract_reply(output, question)
        return self.append_question(reply, question)
    
    def append_question(self, text: str, question: str) -> str:
        """Append the question to the output."""
        return "Question: " + question + "\n" + text

    def extract_reply(self, text: str, question: str) -> str:
        """Extracts the chatbot's reply."""
        if question in text:
            return text.rsplit(question, 1)[-1].strip()
        return text.strip()