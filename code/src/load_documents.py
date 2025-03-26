import os
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma  # Correct import for Chroma
from dotenv import load_dotenv

load_dotenv()

# Initialize Hugging Face embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="product_catalogue",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Directory to save the vector store
)

def load_all_documents_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Iterate over all .txt files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read().strip()  # Remove leading/trailing whitespace
                if not content:
                    print(f"File '{file_path}' is empty. Skipping.")
                    continue
                
                # Create a single Document object for the entire content
                doc = Document(page_content=content, metadata={"source": file_name})
                
                try:
                    # Debug: Log the document being added
                    print(f"Adding the document '{file_name}' to the vector store...")
                    
                    # Add the document to the vector store
                    vector_store.add_documents([doc])
                    
                    # Persist the vector store to disk
                    vector_store.persist()  # Correct way to persist the Chroma vector store
                    print(f"Document '{file_name}' has been successfully added to the vector store.")
                except Exception as e:
                    # Debug: Log the error
                    print(f"Error adding document '{file_name}' to vector store: {e}")
        else:
            print(f"Skipping non-text file or directory: {file_name}")

# Load all documents from the 'doc' folder
load_all_documents_from_folder("doc")