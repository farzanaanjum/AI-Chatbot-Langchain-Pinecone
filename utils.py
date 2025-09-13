from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")  

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get Pinecone API key and environment
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_env = os.environ.get("PINECONE_ENV")
if not pinecone_api_key or not pinecone_env:
    raise ValueError("PINECONE_API_KEY and PINECONE_ENV environment variables not set.")

# Create Pinecone client instance
pc = Pinecone(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

pinecone_index = os.environ.get("PINECONE_INDEX")
if not pinecone_index:
    raise ValueError("PINECONE_INDEX environment variable not set.")
pinecone_index = pinecone_index.strip()

if pinecone_index not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=pinecone_index,
        dimension=512,  # for all-MiniLM-L6-v2 embeddings
        metric='euclidean',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

# Connect to the index (v2.x style)
index = pc.Index(pinecone_index)

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):
    response = openai.chat.completions.create(
    model="gpt-3.5-turbo",  # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"}
    ],
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response.choices[0].message.content

    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    #     temperature=0.7,
    #     max_tokens=256,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )
    # return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

