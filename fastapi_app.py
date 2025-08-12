# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ensure your custom modules are in the Python path.
try:
    from Rag.rag_pipeline_llama import QueryProcessor
except ImportError as e:
    # If the app fails here, it will crash, and the logs will show the error.
    # This is expected for a backend service.
    raise ImportError(f"Could not import QueryProcessor. Check your project structure and PYTHONPATH. Error: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="RAG Query API",
    description="An API to process natural language queries using a RAG pipeline.",
    version="1.0.0",
)

# --- Define Pydantic Model for Request Body ---
# FastAPI uses Pydantic models to automatically validate incoming JSON data.
class QueryRequest(BaseModel):
    question: str

# --- Load the Model ONCE on Startup ---
# FastAPI's `on_event` decorator is the equivalent of Flask's on-startup logic.
# This ensures the model is loaded once when the application starts up.
query_engine = None

@app.on_event("startup")
async def load_model():
    """Load the query engine from disk."""
    global query_engine
    print("Loading query engine...")
    try:
        # NOTE: Make sure the 'storage' directory is accessible.
        query_processor = QueryProcessor()
        query_engine = query_processor  # Assign to the global variable
        print("Query engine loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load the query engine. The application cannot start. Error: {e}")
        # In a production environment, you might use a more robust
        # health check system to indicate failure.

# --- Define the API Route ---
@app.post("/query")
async def handle_query(request_body: QueryRequest):
    """
    Handles a POST request to /query.
    Expects a JSON body like: {"question": "your query here"}
    """
    global query_engine
    if query_engine is None:
        raise HTTPException(status_code=503, detail="Query engine is not available or failed to load.")

    user_query = request_body.question
    print(f"Received query: {user_query}")

    try:
        # Use the pre-loaded engine to get an answer
        answer = query_engine.query(user_query)
        
        # FastAPI automatically handles JSON serialization of the returned dictionary
        return {
            "question": user_query,
            "answer": str(answer)
        }
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the query.")

# --- Main Execution ---
if __name__ == '__main__':
    # This block allows you to run the application directly with `python main.py`.
    # Uvicorn is the ASGI server that runs the FastAPI application.
    # The `reload=True` flag is for local development.
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
