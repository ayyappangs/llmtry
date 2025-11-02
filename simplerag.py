from llama_cpp import Llama
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# RAG config
DATA_DIR = Path("data")
INDEX_DIR = Path("index")
EMB_MODEL = "./Transformers"
TOP_K = 3

# Load or create FAISS index
def build_or_load_faiss():
    if (INDEX_DIR / "faiss.index").exists():
        db = FAISS.load_local(str(INDEX_DIR), HuggingFaceEmbeddings(model_name=EMB_MODEL), allow_dangerous_deserialization=True)
        return db
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        loader = PyMuPDFLoader(str(pdf))
        docs.extend(loader.load())
    if not docs:
        print("No PDFs found in ./data. RAG will not work.")
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    db = FAISS.from_documents(splits, HuggingFaceEmbeddings(model_name=EMB_MODEL))
    db.save_local(str(INDEX_DIR))
    return db

llm = Llama(model_path="Phi-4-mini-reasoning-Q4_K_M.gguf", n_ctx=4096, chat_format="chatml")
db = build_or_load_faiss()
messages = []
print("Welcome to Phi-4 RAG chat! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        print("Goodbye!")
        break
    # RAG: Retrieve relevant context from documents
    context = ""
    if db:
        docs = db.similarity_search(user_input, k=TOP_K)
        context = "\n\n".join([d.page_content for d in docs])
    # Inject context into system prompt
    system_prompt = "You are a helpful assistant. Use the following context to answer the user.\n\n" + context
    chat_msgs = [{"role": "system", "content": system_prompt}] + messages + [{"role": "user", "content": user_input}]
    try:
        resp = llm.create_chat_completion(messages=chat_msgs)
        reply = resp["choices"][0]["message"]["content"]
        print(f"AI: {reply}")
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"Error: {e}")
