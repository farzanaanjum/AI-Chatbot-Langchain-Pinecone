import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
load_dotenv()

def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

if __name__ == "__main__":
    directory = 'data'
    documents = load_docs(directory)
    docs = split_docs(documents)

    # Creating embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    query_result = embeddings.embed_query("Hello world")

    # Storing embeddings in Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pinecone_env = os.environ.get("PINECONE_ENV")
    if not pinecone_api_key or not pinecone_env:
        raise ValueError("PINECONE_API_KEY and PINECONE_ENV must be set in environment variables.")

    pinecone.init(
        api_key=pinecone_api_key,  
        environment=pinecone_env   
    )


    index_name = os.environ.get("PINECONE_INDEX")

    if not index_name:
        raise ValueError("PINECONE_INDEX environment variable not set.")
    index_name = index_name.strip()
    print(f"Using Pinecone index: '{index_name}'")

    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
