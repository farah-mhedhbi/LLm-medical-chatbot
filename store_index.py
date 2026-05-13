from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Step 1 - Load PDF
extracted_data = load_pdf_file("data/")
print(f"Pages loaded: {len(extracted_data)}")

# Step 2 - Split into chunks
text_chunks = text_split(extracted_data)
print(f"Number of chunks: {len(text_chunks)}")

# Step 3 - Load embeddings
embeddings = download_huggingface_embeddings()
print("Embeddings loaded!")

# Step 4 - Create Pinecone index
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

existing_indexes = [i.name for i in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{index_name}' created!")
else:
    print(f"Index '{index_name}' already exists!")

# Step 5 - Store chunks in Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
print("All documents stored in Pinecone successfully!")