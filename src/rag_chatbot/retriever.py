import faiss
from sentence_transformers import SentenceTransformer

class DocsRetriever:

    def __init__(
        self, 
        documents: list[str], 
        model: str = "all-MiniLM-L6-v2"
    ):
        self.documents = documents
        self.model = SentenceTransformer(model)

        embeddings = self.model.encode(documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(
        self, 
        query: str, 
        k: int = 1
    ) -> list[str]:
        query_emb = self.model.encode([query], convert_to_numpy=True)
        _, idcs = self.index.search(query_emb, k)
        return [self.documents[i] for i in idcs[0]]
