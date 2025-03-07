import gradio as gr
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Define constants
CHROMA_PATH = "chroma"

def get_embedding_function():
    embedddings = OllamaEmbeddings(model="nomic-embed-text")
    return embedddings

PROMPT_TEMPLATE = """
If the {question} is a generic conversation like "hi","hello", or similar questions like "how do you do" which do not seek knowledge
from the context, give a natural response without considering the below context. Otherwise answer the question based on the context: {context}.
QUESTION: {question}
ANSWER:
"""

def query_rag(query_text):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="qwen2.5:1.5b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, sources

# Create the Gradio interface
def rag_interface(query):
    response, sources = query_rag(query)
    return response, ", ".join(sources) if sources else "No sources found"

# Define the Gradio app
with gr.Blocks(title="RAG Question Answering System") as demo:
    gr.Markdown("# Retrieval-Augmented Generation (RAG) Q&A System")
    gr.Markdown("Ask a question and the system will retrieve relevant documents and generate an answer.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(label="Your Question", placeholder="Ask something...")
            submit_btn = gr.Button("Submit", variant="primary")
        
    with gr.Row():
        with gr.Column():
            response_output = gr.Textbox(label="Answer")
            sources_output = gr.Textbox(label="Sources")
    
    submit_btn.click(
        fn=rag_interface,
        inputs=query_input,
        outputs=[response_output, sources_output]
    )
    
    gr.Markdown("### How It Works")
    gr.Markdown("""
    1. Your question is processed to find similar documents in the database
    2. The most relevant documents are used as context
    3. An LLM (Qwen2.5 1.5B) generates an answer based on the context
    4. Sources are listed to provide traceability
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()