from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import OPENAI_API_KEY
import os
import openai
import hashlib
import pickle
import time

# Ensure the OpenAI API key used by LangChain/OpenAIEmbeddings matches config.py
# This avoids accidental use of an incorrect environment variable value.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def setup_mental_health_database():
    # Create research_papers directory if it doesn't exist
    if not os.path.exists('research_papers'):
        os.makedirs('research_papers')
        print("Created research_papers directory")
        print("Please add your .txt research papers to the research_papers directory")
        return

    # Check if there are any .txt files
    if not any(f.endswith('.txt') for f in os.listdir('research_papers')):
        print("No .txt files found in research_papers directory")
        print("Please add your research papers as .txt files")
        return

    # Check if cached database exists and is up to date
    cache_file = 'mental_health_db/cached_embeddings.pkl'
    files_hash = calculate_files_hash('research_papers')
    hash_file = 'mental_health_db/files_hash.txt'

    if os.path.exists(cache_file) and os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()

        if stored_hash == files_hash:
            print("Using cached embeddings (no file changes detected)")
            return
        else:
            print("Files changed, rebuilding database...")
    else:
        print("No cache found, building database from scratch...")

    print("Loading research papers...")
    start_time = time.time()
    
    # Load documents with metadata
    documents = []
    for root, _, files in os.walk('./research_papers'):
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    # Get first line as title, or use filename if first line is empty
                    title = content.split('\n')[0].strip() or os.path.splitext(filename)[0]
                    # Create document with metadata
                    from langchain.schema import Document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': title,
                            'filename': filename,
                            'filepath': filepath
                        }
                    )
                    documents.append(doc)
    
    print(f"Loaded {len(documents)} documents in {time.time() - start_time:.2f} seconds")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased from 1000 for fewer chunks
        chunk_overlap=100  # Reduced from 200 for faster processing
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks")

    print("Creating embeddings and storing in database...")
    # Initialize embeddings with the configured API key
    embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    # Create the directory if it doesn't exist
    if not os.path.exists("./mental_health_db"):
        os.makedirs("./mental_health_db")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory="./mental_health_db"
    )
    vectorstore.persist()

    # Save hash of current files
    with open(hash_file, 'w') as f:
        f.write(files_hash)

    # Cache the embeddings
    if not os.path.exists(os.path.dirname(cache_file)):
        os.makedirs(os.path.dirname(cache_file))

    with open(cache_file, 'wb') as f:
        pickle.dump({'chunks': len(splits)}, f)

    print(f"Database created successfully in {time.time() - start_time:.2f} seconds!")

def calculate_files_hash(directory):
    """Calculate a hash based on filenames and modification times"""
    hash_content = ""
    for root, _, files in os.walk(directory):
        for filename in sorted(files):
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                hash_content += f"{filepath}:{mtime}:{size}\n"

    return hashlib.md5(hash_content.encode()).hexdigest()

def get_relevant_documents(query, k=10):  # Increased default k to 10
    """Get relevant documents from the mental health database"""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    # Ensure correct API key is set in environment for embeddings
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Initialize vector database
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma(
        persist_directory="./mental_health_db",
        embedding_function=embeddings
    )
    
    # Search for relevant documents with higher k value and score threshold
    results = db.similarity_search_with_relevance_scores(query, k=k)
    
    # Filter results to only include those with relevance score above threshold
    threshold = 0.5  # Adjust this threshold as needed
    relevant_docs = [doc for doc, score in results if score > threshold]
    
    # If no documents meet the threshold, return at least one document
    if not relevant_docs and results:
        relevant_docs = [results[0][0]]
    
    return relevant_docs

if __name__ == "__main__":
    setup_mental_health_database() 