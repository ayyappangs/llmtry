
# Import the Llama class for running a local LLM (Phi-4 model)
from llama_cpp import Llama
# Path is used for handling file and folder locations
from pathlib import Path
# LangChain document loader for reading PDF files
from langchain_community.document_loaders import PyMuPDFLoader
# Text splitter to break documents into manageable chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
# FAISS is a fast vector search library for similarity search
from langchain_community.vectorstores import FAISS
# Embeddings class to convert text into numerical vectors
from langchain_community.embeddings import HuggingFaceEmbeddings


# ----------- RAG (Retrieval Augmented Generation) configuration -----------
# Folder containing PDF documents to use as knowledge base
DATA_DIR = Path("data")
# Folder to store the FAISS search index
INDEX_DIR = Path("index")
# Path to local embedding model (downloaded offline)
EMB_MODEL = "./Transformers"
# Number of top relevant document chunks to retrieve for each query
TOP_K = 3


# Function to build or load the FAISS vector index for document retrieval
def build_or_load_faiss():
    # If index already exists, load it from disk for fast startup
    if (INDEX_DIR / "faiss.index").exists():
        db = FAISS.load_local(str(INDEX_DIR), HuggingFaceEmbeddings(model_name=EMB_MODEL), allow_dangerous_deserialization=True)
        return db
    # Otherwise, read all PDFs from the data folder
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf))  # Load each PDF
        docs.extend(loader.load())        # Add its pages to the docs list
    # If no documents found, warn user and skip RAG
    if not docs:
        print("No PDFs found in ./data. RAG will not work.")
        return None
    # Split documents into smaller chunks for better retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    # Create FAISS index from document chunks using embeddings
    db = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name=EMB_MODEL))
    db.save_local(str(INDEX_DIR))  # Save index for future use
    return db


# Load the local LLM (Phi-4 model) for answering questions
llm = Llama(model_path="Phi-4-mini-reasoning-Q4_K_M.gguf", n_ctx=4096, chat_format="chatml")
# Build or load the FAISS index for document retrieval
db = build_or_load_faiss()
# Store the conversation history
messages = []

print("Welcome to Phi-4 RAG chat! Type 'exit' to quit.")
while True:
    # Prompt user for input
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    # RAG: Retrieve relevant context from documents using similarity search
    context = ""
    if db:
        docs = db.similarity_search(user_input, k=TOP_K)  # Find top relevant chunks
        context = "\n\n".join([d.page_content for d in docs])  # Combine their text
    # Create a system prompt that includes retrieved context for the LLM
    system_prompt = "You are a helpful assistant. Use the following context to answer the user.\n\n" + context
    # Build the message list for the LLM (system prompt + chat history + user input)
    chat_msgs = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}]
    try:
        # Ask the LLM to generate a response using the context and conversation
        resp = llm.create_chat_completion(messages=chat_msgs)
        reply = resp["choices"][0]["message"]["content"]
        print(f"AI: {reply}")
        # Add user and assistant messages to the conversation history
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        # Print any errors that occur during response generation
        print(f"Error: {e}")
