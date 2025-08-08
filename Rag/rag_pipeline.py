import warnings
from docs.document_processor import DocumentProcessor
from chains.qa_chain import QAChainBuilder

warnings.filterwarnings("ignore", category=UserWarning)


class RAGPipeline:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vectorstore = None
        self.qa_chain = None

    def prepare_documents(self, pdf_path: str):
        """
        Load and split the PDF, then create the vectorstore.
        """
        split_docs = self.doc_processor.load_and_split_pdf(pdf_path)
        self.doc_processor.create_vectorstore(split_docs)

    def load_vectorstore(self):
        """
        Load the persisted vectorstore.
        """
        self.vectorstore = self.doc_processor.load_vectorstore()

    def ask_question(self, question: str) -> str:
        """
        Build the QA chain and ask the question.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not loaded. Call load_vectorstore() first.")

        qa_builder = QAChainBuilder(self.vectorstore)
        self.qa_chain = qa_builder.build_chain()
        response = self.qa_chain.invoke(question)
        return response['result']


