import streamlit as st
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Define constants
CHROMA_PATH = "chroma"

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

PROMPT_TEMPLATE = """
If the {question} is a generic conversation like "hi", "hello", or similar questions like "how do you do" 
which do not seek knowledge from the context, give a natural response without considering the below context. 
Otherwise, answer the question based on the context: {context}.

QUESTION: {question}
ANSWER:
"""

def query_rag(query_text):    
    # Step 1: Load Embeddings and Vector Store Retriever
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 similar docs

    # Step 2: Define Prompt Template
    PROMPT_TEMPLATE = """
    If the {question} is a generic conversation like "hi", "hello", or similar small talk, 
    give a natural response without using the context. Otherwise, answer the question using the retrieved context:

    CONTEXT:
    {context}

    QUESTION: {question}
    ANSWER:
    """
    chatprompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    # Step 3: Define LLM
    llm = Ollama(model="qwen2.5:1.5b")

    # Step 4: Define Chain
    chain = (
        {
            "context": retriever,  # Retrieve relevant documents
            "question": RunnablePassthrough(),  # Pass the question directly
        }
        | chatprompt  # Format input into a structured prompt
        | llm  # Generate response using the LLM
        | StrOutputParser()  # Extract string output
    )

    # Step 5: Invoke the Chain    
    response = chain.invoke(query_text)
    return response
# Streamlit UI
st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) Q&A System")
st.write("Ask a question, and the system will retrieve relevant documents and generate an answer.")

query = st.text_input("Your Question", placeholder="Ask something...")

if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            response = query_rag(query)
        
        st.subheader("Answer")
        st.write(response)
                
    else:
        st.warning("Please enter a question.")

st.markdown("### How It Works")
st.markdown("""
1. Your question is processed to find similar documents in the database.
2. The most relevant documents are used as context.
3. An LLM (Qwen2.5 1.5B) generates an answer based on the context.
""")

