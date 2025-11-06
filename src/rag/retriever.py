from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import os
from typing import List

load_dotenv()

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(
    "../Data/vector_store", embeddings, allow_dangerous_deserialization=True
)


def retrieve_from_vector_store(queries: List[str] | str, k: int = 3) -> List[Document]:
    """Return relevant document chunks for one or more queries.

    Args:
        queries (List[str] | str): A single query string or a list of query strings.
        k (int, optional): The number of closest chunks per query. Defaults to 10.

    Returns:
        List[Document]: Combined list of retrieved document chunks across all queries.
    """
    if isinstance(queries, str):
        queries = [queries]

    all_docs = []
    for query in tqdm(queries, desc="Retrieving documents"):
        try:
            results = db.similarity_search(query, k=k)
            all_docs.extend(results)
        except Exception as e:
            print(f"Error retrieving for query '{query}': {e}")
            continue

    return all_docs
