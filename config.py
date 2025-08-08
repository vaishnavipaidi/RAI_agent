from langchain_openai  import AzureChatOpenAI
from dotenv import load_dotenv

import os
 
load_dotenv()
 
llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            streaming=True
        )
 
# print(llm.invoke("what is RAI"))