# from langchain_openai  import AzureChatOpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from dotenv import load_dotenv

import os
 
load_dotenv()
 
# llm = AzureChatOpenAI(
#             deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             api_version=os.getenv("AZURE_API_VERSION"),
#             streaming=True
#         )


llm = AzureOpenAI(
    model="gpt-4.1",
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    streaming=True
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
)


 
# print(llm.invoke("what is RAI"))