import os
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
# from llama_index.storage.storage_context import StorageContext
# from llama_index.text_splitter import SentenceSplitter
import logging

logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()