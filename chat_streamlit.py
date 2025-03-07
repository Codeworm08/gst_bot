import streamlit as st
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama

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
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="qwen2.5:1.5b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _ in results]
    return response_text, sources

# Streamlit UI
st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("Retrieval-Augmented Generation (RAG) Q&A System")
st.write("Ask a question, and the system will retrieve relevant documents and generate an answer.")

query = st.text_input("Your Question", placeholder="Ask something...")

if st.button("Submit"):
    if query:
        with st.spinner("Generating response..."):
            response, sources = query_rag(query)
        
        st.subheader("Answer")
        st.write(response)
        
        st.subheader("Sources")
        st.write(", ".join(sources) if sources else "No sources found")
    else:
        st.warning("Please enter a question.")

st.markdown("### How It Works")
st.markdown("""
1. Your question is processed to find similar documents in the database.
2. The most relevant documents are used as context.
3. An LLM (Qwen2.5 1.5B) generates an answer based on the context.
4. Sources are listed to provide traceability.
""")

