import os
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings, ChatCohere, CohereRerank, CohereRagRetriever
from langchain.text_splitter import CharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from langchain_community.vectorstores import Chroma
import pickle

os.environ["COHERE_API_KEY"] = "fXM87G3LCf0twlAHpyos3wOqzWfKNnZKzv9xnQ0C"

# Define the Cohere LLM
llm = ChatCohere(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="command-r-plus-08-2024"
)

# Define the Cohere embedding model
embeddings = CohereEmbeddings(
    cohere_api_key=os.environ["COHERE_API_KEY"],
    model="embed-english-light-v3.0"
)

VECTOR_STORE_PATH = "chroma_vector_store"

def setup_vectorstore(documents):
    """Set up the vector store and return the retriever."""
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(documents)

    # Create a vector store from the documents
    db = Chroma.from_documents(split_documents, embeddings, persist_directory=VECTOR_STORE_PATH)
    db.persist()

    return db

def load_document():
    """Load documents from the predefined directory."""
    knowledge_base_path = "tmp/"  # Set your predefined path here
    reader = SimpleDirectoryReader(input_dir=knowledge_base_path, exclude=["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"])
    documents = reader.load_langchain_documents()
    print(documents[:10])
    return documents

def create_chain(db, user_query):
    """Create the conversation chain using the retriever."""
    # Create Cohere's reranker with the vector DB
    reranker = CohereRerank(
        cohere_api_key=os.environ["COHERE_API_KEY"],
        model="rerank-english-v3.0"
    )

    # Contextual Compression Retriever
    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=db.as_retriever()
    )
    compressed_docs = retriever.get_relevant_documents(user_query)

    # Create the Cohere RAG retriever
    rag = CohereRagRetriever(llm=llm, connectors=[])
    docs = rag.get_relevant_documents(user_query, documents=compressed_docs)

    # Append query and answer to chat history
    answer = docs[-1].page_content
    return docs, answer

# Streamlit App
st.title("Knowledge Base Query")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None



# Button to load knowledge base
if st.button("Load Knowledge Base"):
    with st.spinner("Loading documents..."):
        st.session_state.raw_documents = load_document()
        st.session_state.vectorstore = setup_vectorstore(st.session_state.raw_documents)
        st.success("Documents loaded successfully!")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and query processing
user_input = st.chat_input("Ask Llama...")

if user_input:
    # Append user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query and generate response
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            docs, assistant_response = create_chain(st.session_state.vectorstore, user_input)
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
