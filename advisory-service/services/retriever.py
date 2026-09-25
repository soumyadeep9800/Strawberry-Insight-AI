from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent

VECTOR_STORE_DIR = BASE_DIR / "vector_store"


# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load existing ChromaDB
vector_store = Chroma(
    persist_directory=str(VECTOR_STORE_DIR),
    collection_name="strawberry_diseases",
    embedding_function=embeddings
)


def retrieve_information(
    disease: str,
    query: str,
    k: int = 5
):

    results = vector_store.similarity_search(
        query,
        k=k,
        filter={
            "disease": disease
        }
    )

    return results