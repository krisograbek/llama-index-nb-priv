import os

from typing import Sequence
from llama_index.core.schema import BaseNode

from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter

from llama_index.vector_stores.qdrant import QdrantVectorStore


from dotenv import load_dotenv
from qdrant_client import qdrant_client

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

qdrantClient = qdrant_client.QdrantClient(
    location=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


# directory containing the documents to index
DATA_DIR = "/home/kris/dev/syenza-docs/pdfs"
# INDEX_NAME = "mrvs-kris-v1"

from typing import Sequence
from llama_index.core.schema import BaseNode

def documents_to_nodes(documents, chunk_size: int = 1024, chunk_overlap: int = 200) -> Sequence[BaseNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    nodes = splitter.get_nodes_from_documents(documents)
    
    return nodes


def create_index(nodes, colection_name):
    vector_store = QdrantVectorStore(colection_name, client=qdrantClient)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)


def inject_docs(documents, chunk_size, chunk_overlap):
    print("Creating nodes:")
    nodes = documents_to_nodes(documents, chunk_size, chunk_overlap)
    collection_name = "mrvs-chunks-{}-{}".format(chunk_size, chunk_overlap)
    print(f"Generating index for collection: {collection_name}")
    print(f"Generated {len(nodes)} nodes of type: {type(nodes[0])}")
    create_index(nodes, collection_name) 

sizes_and_overlaps = [
    # (1024,100), # check
    (1024,200), # check but only 26 files...
    (512,100), # FAIL
    # (512,50), # check
    # (256,50), # check
    # (256,25) # check
]

print(f"Loading the documents from: {DATA_DIR}")
documents = SimpleDirectoryReader(DATA_DIR).load_data()

for cs, co in sizes_and_overlaps:
    print(f"Running for Chunk size: {cs}, and overlap: {co}")
    inject_docs(documents, cs, co)
    print("---"*25)


# service_context = ServiceContext.from_defaults(
#     llm=OpenAI(model="gpt-4-1106-preview", api_key=api_key),
#     system_prompt=expert_system_prompt,
# )


# vector_store = QdrantVectorStore(client=qdrantClient, collection_name=INDEX_NAME)
# # check if storage already exists

# documents = SimpleDirectoryReader(DATA_DIR).load_data()

# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# VectorStoreIndex.from_documents(
#     documents,
#     storage_context=storage_context,
# )
