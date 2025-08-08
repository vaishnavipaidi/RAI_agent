from Rag.rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Step 1: Prepare documents and create vectorstore (only needed once)
# pipeline.prepare_documents("docs/responsible-ai.pdf")

# Step 2: Load vectorstore
pipeline.load_vectorstore()

# Step 3: Ask your question
question = """Effective internal governance relies on key organizational controls...
what are Other essential elements?"""

answer = pipeline.ask_question(question)

print("\n[Answer]")
print(answer)
