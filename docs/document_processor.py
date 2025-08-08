# docs/document_processor.py
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import nltk
# nltk.download("averaged_perceptron_tagger_eng")


class DocumentProcessor:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

    def load_and_split_pdf(self, pdf_path):
        print(f"the path of pdf: {pdf_path}")

        # loader = PyPDFLoader(pdf_path)
        loader = UnstructuredPDFLoader(pdf_path, mode="elements")
        documents = loader.load()
        for i, doc in enumerate(documents[:50]):
            print(f"\n--- Document {i+1} ---\n")
            print(doc.page_content[:2000])
        if not documents:
            raise ValueError(f"No Documents loaded from: {pdf_path}")
        split_docs = self.text_splitter.split_documents(documents)
        print("splitted docs chunks are:",split_docs)
        if not split_docs:
            raise ValueError(f"No content to split in: {pdf_path}")
        return split_docs

    def create_vectorstore(self, split_docs, persist_path: str = "vectorstore_unstructured/"):
        if not split_docs:
            raise ValueError("No split documents provided to create vectorstore.")
        os.makedirs(persist_path, exist_ok=True)
        vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        vectorstore.save_local(persist_path)
        os.makedirs(persist_path, exist_ok=True)
        print("created a vector store")
        return vectorstore

    def load_vectorstore(self, persist_path="vectorstore_unstructured"):
        return FAISS.load_local(persist_path, self.embeddings, allow_dangerous_deserialization=True)
    



