# streamlit_app.py
import streamlit as st
# Ensure your custom modules are in the Python path.
# You might need to adjust the import paths depending on your project structure.
try:
    from docs.document_processor_llama import EmbeddingGenerator
    from Rag.rag_pipeline_llama import QueryProcessor
except ImportError as e:
    st.error(f"Failed to import a required module: {e}")
    st.error("Please make sure the 'docs' and 'Rag' directories are in the same folder as this Streamlit app, or adjust the system path.")
    st.stop()


# --- Important Note on Embeddings ---
# The following function to create embeddings is a one-time setup process.
# You should run this from a separate Python script *before* you launch
# the Streamlit app to avoid re-creating the index every time.
#
# --- SCRIPT TO RUN ONCE: create_embeddings.py ---
# from docs.document_processor_llama import EmbeddingGenerator
# print("Creating and saving embeddings...")
# embedder = EmbeddingGenerator(pdf_dir="docs/responsible-ai.pdf")
# embedder.create_and_save_index()
# print("Done.")


# We use st.cache_resource to load the query engine only once.
# This prevents reloading the model every time the user asks a question,
# which makes the app much faster.
@st.cache_resource
def load_query_engine():
    """Loads the QueryProcessor and returns the query engine."""
    try:
        query_processor = QueryProcessor()
        return query_processor
    except Exception as e:
        st.error(f"Failed to load the query engine. Make sure the 'storage' directory with your index exists. Error: {e}")
        return None

def get_answer(query_engine, query):
    """Queries the engine and returns the answer."""
    if query_engine is None:
        return "Error: Query engine is not available."
    
    answer = query_engine.query(query)
    return answer

# --- Streamlit User Interface ---
st.set_page_config(page_title="Document Q&A", layout="wide")

st.title("📄 Document Q&A with LlamaIndex")
st.markdown("This app uses a RAG pipeline to answer questions about the content of a pre-processed PDF document ('responsible-ai.pdf').")

# Load the query engine
query_engine = load_query_engine()

if query_engine:
    # Get user input from a text box
    user_query = st.text_input(
        "Ask a question about the document:", 
        placeholder="e.g., what are the Evolution of Autonomy"
    )

    # Create a button to submit the query
    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Searching for the answer in the document..."):
                # Call the main logic function
                answer = get_answer(query_engine, user_query)
                
                # Display the answer
                st.success("Answer found!")
                st.markdown("### 💬 Answer:")
                st.write(str(answer)) # Using str() to ensure it's displayable
        else:
            st.warning("Please enter a question.")
else:
    st.error("The application could not start. Please check the console for errors.")

