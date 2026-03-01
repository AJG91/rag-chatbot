import faiss
from sentence_transformers import SentenceTransformer

class DocsRetriever:
    """
    A class used to retrieve information from documents similar to a query.
    
    Attributes
    ----------
    documents : list[str]
    model : str
    index : faiss.Index
        Constructs an exact FAISS index using squared L2 distance.

    Parameters
    ----------
    documents : list[str]
        List of text that will be encoded and retrieved.
    model : str, optional (default="all-MiniLM-L6-v2")
        Name of model that will be loaded.
    """
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
        """
        Retrieves the top k most similar documents in the query string.

        Parameters
        ----------
        query : str
            Natural language query to embed and search against the index.
        k : int, optional (default=1)
            Number of nearest documents to return. Must be >= 1.

        Returns
        -------
        list[str]
            List of retrieved documents ordered by increasing L2 distance         (i.e., most similar first).
        """
        query_emb = self.model.encode([query], convert_to_numpy=True)
        _, idcs = self.index.search(query_emb, k)
        return [self.documents[i] for i in idcs[0]]
