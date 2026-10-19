"""Load the three placement knowledge files and persist their chunks in ChromaDB."""
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "campus_placement_knowledge"
EXPECTED_FILES = [
    "placement_eligibility.txt",
    "placement_registration.txt",
    "placement_conduct.txt",
]

def main():
    missing = [name for name in EXPECTED_FILES if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing knowledge document(s): " + ", ".join(missing) +
            ". Put all three .txt files in the data/ folder."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing. Add it to your .env file.")

    documents = []
    for filename in EXPECTED_FILES:
        path = DATA_DIR / filename
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{filename} is empty.")
        documents.append(Document(page_content=text, metadata={"source": filename}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    # Rebuild the collection so repeated ingestion cannot accumulate duplicate chunks.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Ingestion complete: {len(documents)} documents -> {len(chunks)} chunks.")
    print(f"ChromaDB stored at: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
