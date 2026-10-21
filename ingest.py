"""Load the three placement knowledge files and persist their chunks in ChromaDB."""
import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency for local env files
    def load_dotenv(*_args, **_kwargs):
        return False

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def _get_api_key() -> str | None:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")


# Import the embedding class after the API key check so the script fails with a clear message
# when the required provider configuration is missing.
def _build_embeddings():
    google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"))

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))

    raise EnvironmentError(
        "No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file before running ingestion."
    )


def main():
    missing = [name for name in EXPECTED_FILES if not (DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing knowledge document(s): " + ", ".join(missing) +
            ". Put all three .txt files in the data/ folder."
        )

    if not _get_api_key():
        raise EnvironmentError(
            "No API key found. Add GOOGLE_API_KEY or OPENAI_API_KEY to your .env file before running ingestion."
        )

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
        try:
            shutil.rmtree(CHROMA_DIR)
        except PermissionError as exc:
            raise RuntimeError(
                f"Cannot delete the ChromaDB directory: {CHROMA_DIR}\n"
                "close all applications using ChromaDB, then try again."
            ) from exc

    embeddings = _build_embeddings()
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
