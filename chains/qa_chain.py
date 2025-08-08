# chains/qa_chain.py
from langchain.chains import RetrievalQA
from config import llm

class QAChainBuilder:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def build_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain
