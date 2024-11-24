#!/usr/bin/env python3
import os
import time
import gradio as gr
from groq import Groq
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate

# Environment variables
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Groq client
client = Groq()

def get_groq_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from Groq API"""
    try:
        full_response = ""
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="llama3-8b-8192",
            stream=True,
        )
        
        # Properly handle streaming response
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                
        return full_response
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return f"Error: {str(e)}"

def setup_retriever():
    """Set up the document retriever"""
    try:
        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        
        # Ensure persist directory exists
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Load Vector Store
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY, 
            embedding_function=embeddings
        )
        
        # Return retriever
        return db.as_retriever(search_kwargs={"k": 4})
    
    except Exception as e:
        print(f"Error setting up retriever: {e}")
        raise

def process_query(query: str, show_sources: bool = False) -> str:
    """Process the user's query"""
    if not query.strip():
        return "Please enter a query."
    
    try:
        # Get retriever
        retriever = setup_retriever()
        
        # Use invoke instead of get_relevant_documents
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        start = time.time()
        
        # System prompt
        system_prompt = """You are QLIPpy, a helpful, informative, and concise chatbot to help users understand what QLIP does and how it can help them.
        The people you will speak to will likely be youth workers and practitioners who work with young people.
        
        QLIP (Quality, Leadership and Impact Partnership) is an initiative led by London Youth and Mary's Youth Club, commissioned by Islington Council. 
        It provides wrap-around support to youth organisations and practitioners in the borough through professional development activities.
        
        Guidelines:
        1. Use the provided context to answer questions.
        2. If you don't know something, be honest and provide contact information for the team so they can take the query forward.
        3. Always link back to the relevant page on the website (qlip.org.uk) for the user to find out more information.
        4. Keep your answers short but informative. Use a maximum of four sentences.
        5. If possible, suggest related questions the user might want to ask.
        6. Format your response so it's well structured and spaced out, and is in Markdown format.
        
        Contact information:
        - For queries regarding training, supervision, and first aid, contact Sally Baxter, Mary's CEO & Youth Development Manager, at sally.baxter@marys.org.uk.
        - For queries regarding the MEL Framework, and bespoke support, contact Ruth Virgo, London Youth's Youth Sector Development Manager, at ruth.virgo@londonyouth.org.
        - We also send out a monthly newsletter to keep you and the sector up to date with what we do. Subscribe hear about: Updates about the youth sector, upcoming funded training, curated funding opportunities, and our impact report and data collection
          Subscribe here: https://mailchi.mp/ba929b010b40/qlip-newsletter
        """
        
        # User prompt with context
        user_prompt = f"""Based on this context:
        {context}
        
        Please answer this question:
        {query}"""
        
        # Get response from Groq
        answer = get_groq_response(system_prompt, user_prompt)
        
        # Add sources if requested
        if show_sources:
            sources_text = "\n\nSources:\n"
            for doc in docs:
                sources_text += f"\nFrom {doc.metadata.get('source', 'Unknown Source')}:\n{doc.page_content}\n"
            answer += sources_text
        
        end = time.time()
        
        return f"{answer}\n\nQuery processed in {end - start:.2f} seconds"
    
    except Exception as e:
        return f"Error processing query: {e}"

def create_gradio_interface():
    """Create the Gradio interface for the application"""
    with gr.Blocks(title="QLIPpy") as interface:
        gr.Markdown("# QLIPpy")
        gr.Markdown("Ask questions about QLIP.")
        
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="Type your question here...",
                    lines=3
                )
                
                show_sources = gr.Checkbox(
                    label="Show source documents",
                    value=False
                )
                
                submit_btn = gr.Button("Submit")
            
            with gr.Column():
                output = gr.Markdown()

        submit_btn.click(
            fn=process_query,
            inputs=[query_input, show_sources],
            outputs=output
        )
        
    return interface

def main():
    # Create and launch the Gradio interface
    interface = create_gradio_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()