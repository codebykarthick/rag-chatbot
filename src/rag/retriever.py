from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from typing import List

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(
    "../../Data/vector_store", embeddings, allow_dangerous_deserialization=True
)


def retrieve_from_vector_store(query: str, k: int = 3) -> List[Document]:
    """Return the relevant document chunks for the given query based on
    match with embeddings.

    Args:
        query (str): The query posted by the user
        k (int, optional): The number of closest chunks that we have to retrieve. Defaults to 3.

    Returns:
        _type_: _description_
    """
    return db.similarity_search(query, k=k)
