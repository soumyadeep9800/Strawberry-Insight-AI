from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base" / "diseases"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"


# --------------------------------------------------
# Load Markdown files
# --------------------------------------------------

def load_documents():
    loader = DirectoryLoader(
        str(KNOWLEDGE_BASE_DIR),
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    return documents


# --------------------------------------------------
# Split documents into chunks
# --------------------------------------------------

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:

        source = chunk.metadata.get("source", "")

        filename = Path(source).stem

        disease_names = {
            "leaf_spot": "Leaf Spot",
            "powdery_mildew_leaf": "Powdery Mildew Leaf",
            "gray_mold": "Gray Mold",
            "angular_leafspot": "Angular Leafspot",
            "blossom_blight": "Blossom Blight",
            "powdery_mildew_fruit": "Powdery Mildew Fruit",
            "anthracnose_fruit_rot": "Anthracnose Fruit Rot",
        }

        chunk.metadata["disease"] = disease_names.get(
            filename,
            filename
        )

    print(f"Created {len(chunks)} chunks")

    return chunks


# --------------------------------------------------
# Create embeddings
# --------------------------------------------------

def create_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# --------------------------------------------------
# Create ChromaDB
# --------------------------------------------------

def create_vector_store():

    documents = load_documents()

    chunks = split_documents(documents)

    embeddings = create_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name="strawberry_diseases"
    )

    print("ChromaDB vector store created successfully!")

    return vector_store


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    create_vector_store()