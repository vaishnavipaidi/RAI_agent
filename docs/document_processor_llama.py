# docs/document_processor_llama.py
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.readers.llamaparse import LlamaParseReader
from llama_parse import LlamaParse
import faiss
import os
from config import llm, embed_model
from dotenv import load_dotenv
load_dotenv()

class EmbeddingGenerator:
    def __init__(self, pdf_dir: str, faiss_index_path: str = "vector_index"):
        self.pdf_dir = pdf_dir
        self.faiss_index_path = faiss_index_path
        self.embed_model = embed_model

    def load_documents(self):
        parser = LlamaParse(api_key=os.getenv("llama_parse_key"),  # Replace with your actual API key
                            result_type="text",  # or "text"
                            verbose=True, # optional: set to False to reduce output
                            language="en" # optional: default is "en"
        )
        documents = parser.load_data(self.pdf_dir)
        print("documents parsed")
        return documents

    def create_and_save_index(self):
        documents = self.load_documents()
        dimension = 1536  # For Azure OpenAI's embedding model like `text-embedding-ada-002`
        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=self.embed_model)

        # save index
        os.makedirs(self.faiss_index_path, exist_ok=True)
        index.storage_context.persist(self.faiss_index_path)
        print("✅ FAISS index saved at", self.faiss_index_path)
