import argparse
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "chroma"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
print("OpenAI API Key:", os.getenv('OPENAI_API_KEY'))
# App framework
st.title('🦜🔗 Alice in wonder world')
question = st.text_input('Plug in your prompt here') 

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

query_text = question
# st.write("working")

# Prepare the DB.
embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
# st.write("done embeddings")

# Search the DB.
results = db.similarity_search_with_relevance_scores(query_text, k=3)
if len(results) == 0 or results[0][1] < 0.7:
    print(f"Unable to find matching results.")

# st.write("found match")
context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)

model = ChatOpenAI()
response_text = model.predict(prompt)

sources = [doc.metadata.get("source", None) for doc, _score in results]
formatted_response = f"Response: {response_text}\nSources: {sources}"
print(formatted_response)
st.write(formatted_response)
