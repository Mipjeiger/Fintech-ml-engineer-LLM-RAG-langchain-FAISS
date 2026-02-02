"""
Script to create vector database - Run this instead of slow notebook cell
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("🔄 Loading PDF...")
# Fixed path - PDF is in pdf/ subfolder
loader = PyPDFLoader("./pdf/guidelines-for-implementing-anti-fraud-strategies-v2.pdf")
docs = loader.load()

print(f"📄 Loaded {len(docs)} pages")

print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Clean metadata
for c in chunks:
    c.metadata = {
        "source": c.metadata.get("source", ""),
        "page": c.metadata.get("page", 0),
    }

print(f"✅ Created {len(chunks)} chunks")

print("🤖 Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
)

print("🏗️  Creating vector database (this will take 2-3 minutes)...")
vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)

print("💾 Saving vector database...")
vectordb.save_local("./fraud_vectordb")

print(f"\n✅ SUCCESS! Vector database created with {vectordb.index.ntotal} documents")
print("✅ Saved to ./fraud_vectordb")
print("\n📝 Now you can load it in your notebook with:")
print(
    "   vectordb = FAISS.load_local('./fraud_vectordb', embeddings, allow_dangerous_deserialization=True)"
)
