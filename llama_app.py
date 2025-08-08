#llama_app.py
from docs.document_processor_llama import EmbeddingGenerator
from Rag.rag_pipeline_llama import QueryProcessor

def main(query):
    # Step 1: Create and save embeddings
    # embedder = EmbeddingGenerator(pdf_dir="docs/responsible-ai.pdf")
    # embedder.create_and_save_index()

    # Step 2: Ask questions
    query_engine = QueryProcessor()

    answer = query_engine.query(query)
    return answer
    

if __name__ == "__main__":
    user_input="what are the key points for HCLtech Responsible AI tenets?"
    answer=main(user_input)
    print("💬 Answer:", answer)
