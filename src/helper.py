from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List
from langchain.schema import Document
import os
import requests


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []
    for doc in docs:
        minimal_doc = Document(
            page_content=doc.page_content,
            metadata=doc.metadata
        )
        minimal_docs.append(minimal_doc)
    return minimal_docs


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def download_huggingface_embeddings():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    # Test the token first
    test_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    test_response = requests.post(
        test_url,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": "test", "options": {"wait_for_model": True}}
    )
    print(f"HuggingFace API status: {test_response.status_code}")
    print(f"HuggingFace API response: {test_response.text[:200]}")
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )