from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings.base import Embeddings
import os
import requests


class HuggingFaceAPIEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": texts}
        )
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json={"inputs": [text]}
        )
        return response.json()[0]


def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


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
    return text_splitter.split_documents(extracted_data)


def download_huggingface_embeddings():
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    print(f"Token: {'Found' if token else 'NOT FOUND'}")
    embeddings = HuggingFaceAPIEmbeddings(
        api_key=token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings