import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

print("Step 1: Loading document...")
loader = TextLoader("data/faq.txt", encoding="utf-8")
documents = loader.load()
print(f"Documents loaded: {len(documents)}")

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

print("Step 3: Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model ready!")

print("Step 4: Saving to ChromaDB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
print("Done! chroma_db folder created.")
print("Ingest complete! Now run app.py")