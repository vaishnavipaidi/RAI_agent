# Rag/rag_pipline_llama.py
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from config import llm, embed_model

class QueryProcessor:
    def __init__(self, faiss_index_path="vector_index"):
        self.faiss_index_path = faiss_index_path
        self.embed_model = embed_model
        self.llm=llm

    def load_index(self):
        vector_store = FaissVectorStore.from_persist_dir(self.faiss_index_path)
        storage_context = StorageContext.from_defaults(persist_dir=self.faiss_index_path, vector_store=vector_store)
        index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        return index

    def query(self, question: str, top_k: int = 3):
        index = self.load_index()
        query_engine = index.as_query_engine(similarity_top_k=top_k,
                                             llm=self.llm
                                             )
        response = query_engine.query(question)
        return str(response)
