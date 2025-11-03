from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import glob
import os

# Load config for preprocessing
load_dotenv()
CHUNK_SIZE = int(os.environ["CHUNK_SIZE"])
CHUNK_OVERLAP = int(os.environ["CHUNK_OVERLAP"])

# Recursively find all PDF and DOCX files under the data directory
file_paths = glob.glob("../Data/**/*", recursive=True)
file_paths = [f for f in file_paths if f.lower().endswith((".pdf", ".docx"))]

# Load all content textual content into the list
docs = []
for path in file_paths:
    if path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(path)
    elif path.lower().endswith(".docx"):
        loader = Docx2txtLoader(path)
    else:
        continue
    docs.extend(loader.load())

# Chunk all docs
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(docs)

# Generate embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector db from all chunks and their corresponding embeddings
# and store them to prevent repetition (Unless needed.)
db = FAISS.from_documents(chunks, embeddings)
db.save_local("../Data/vector_store")
print("Vector store saved to Data/vector_store")
