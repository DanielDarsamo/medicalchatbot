import os
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone

# Load environment variables
load_dotenv()

# Get Pinecone API key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please check your .env file.")

# Import your custom helper functions
from src.helper import load_pdf_file, text_split

try:
    # Extract data and split into chunks
    extracted_data = load_pdf_file(data='Data/')
    text_chunks = text_split(extracted_data)

    # Use updated HuggingFace Embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # Initialize Pinecone client
    pc = PineconeClient(api_key=PINECONE_API_KEY)

    # Define index name
    index_name = "medicalbot"

    # Create index if it doesn't exist
    try:
        pc.create_index(
            name=index_name,
            dimension=384,  # Confirm this matches your embedding dimension
            metric="cosine", 
            spec=ServerlessSpec(
                cloud="aws", 
                region="us-east-1"
            ) 
        )
    except Exception as index_error:
        print(f"Index may already exist or there was an error: {index_error}")

    # Upsert documents into Pinecone index
    docsearch = Pinecone.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print("Indexing completed successfully.")

except Exception as e:
    print(f"An error occurred: {e}")