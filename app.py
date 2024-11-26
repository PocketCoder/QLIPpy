# Fix for SQLite version - MUST be at the very top before other imports
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import time
import streamlit as st
from groq import Groq
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from PIL import Image

# Environment variables and constants
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Groq client
client = Groq()

# Streamlit page config
st.set_page_config(
    page_title="QLIPpy",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for better markdown formatting
st.markdown("""
    <style>
        .stMarkdown {
            font-size: 1.1rem;
        }
        .stMarkdown h1 {
            color: #1E3A8A;
        }
        .stMarkdown h2 {
            color: #2563EB;
        }
        .stMarkdown h3 {
            color: #3B82F6;
        }
        .stMarkdown a {
            color: #2563EB;
            text-decoration: underline;
        }
        .stMarkdown blockquote {
            border-left: 3px solid #3B82F6;
            padding-left: 1rem;
            color: #4B5563;
        }
        .stMarkdown code {
            padding: 0.2em 0.4em;
            background-color: #F3F4F6;
            border-radius: 3px;
        }
        .response-container {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            margin: 1rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# Load and display logo
logo_path = 'static/logo.png'
logo = Image.open(logo_path)

def get_groq_response(system_prompt: str, user_prompt: str) -> str:
    """Get response from Groq API with streaming"""
    try:
        # Create a container for the response with styling
        response_container = st.container()
        with response_container:
            # Add the response-container class
            st.markdown('<div class="response-container">', unsafe_allow_html=True)
            response_placeholder = st.empty()
        
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
        
        # Stream the response with markdown rendering
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                # Update the response in real-time with markdown rendering
                response_placeholder.markdown(full_response + "▌")
        
        # Final update without cursor
        response_placeholder.markdown(full_response)
        # Close the container div
        st.markdown('</div>', unsafe_allow_html=True)
        
        return full_response
    
    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return f"Error: {str(e)}"

@st.cache_resource
def setup_retriever():
    """Set up the document retriever with caching"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Updated Chroma initialization without explicit config
        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        return db.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        raise

def process_query(query: str, show_sources: bool = False) -> str:
    """Process the user's query"""
    if not query.strip():
        return "Please enter a query."
    
    try:
        retriever = setup_retriever()
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        start = time.time()
        
        system_prompt = """You are QLIPpy, a helpful, informative, and concise chatbot to help users understand what QLIP does and how it can help them.
        The people you will speak to will likely be youth workers and practitioners who work with young people.
        
        QLIP (Quality, Leadership and Impact Partnership) is an initiative led by London Youth and Mary's Youth Club, commissioned by Islington Council. 
        It provides wrap-around support to youth organisations and practitioners in the borough through professional development activities.
        
        Guidelines:
        1. Use the provided context to answer questions.
        2. If you don't know something, be honest and provide contact information for the team so they can take the query forward. Do not hallucinate linkes, services, or other aspects of the QLIP program.
        3. Always link back to the relevant page on the website (qlip.org.uk) for the user to find out more information.
        4. Keep your answers short but informative. Use a maximum of four sentences.
        5. If possible, suggest related questions the user might want to ask.
        6. Format your response using Markdown:
           - Use ## for main headings
           - Use ### for subheadings
           - Use **bold** for emphasis
           - Use `code` for specific terms or references
           - Use > for important quotes or highlights
           - Use bullet points (*) and new lines for lists
           - Format links properly: [text](url)
        
        Contact information:
        - For queries regarding training, supervision, and first aid, contact Sally Baxter, Mary's CEO & Youth Development Manager, at sally.baxter@marys.org.uk.
        - For queries regarding the MEL Framework, and bespoke support, contact Ruth Virgo, London Youth's Youth Sector Development Manager, at ruth.virgo@londonyouth.org.
        - We also send out a monthly newsletter to keep you and the sector up to date with what we do.
          Subscribe here: https://mailchi.mp/ba929b010b40/qlip-newsletter
        """
        
        user_prompt = f"""Based on this context:
        {context}
        
        Please answer this question:
        {query}"""
        
        answer = get_groq_response(system_prompt, user_prompt)
        
        end = time.time()
        processing_time = f"\n\n*Query processed in {end - start:.2f} seconds*"
        
        if show_sources:
            sources_text = "\n\n## Sources\n"
            for doc in docs:
                sources_text += f"\n### From {doc.metadata.get('source', 'Unknown Source')}\n{doc.page_content}\n"
            return answer + processing_time + sources_text
        
        return answer + processing_time
    
    except Exception as e:
        st.error(f"Error processing query: {e}")
        return f"Error processing query: {e}"

def main():
    st.title("QLIPpy")
    st.markdown("Ask questions about QLIP.")
    st.markdown("""
    *Your first question will take a few seconds to load, but after that it should be quick!*

    *We have a limit of 30 questions a minute. If it doesn't work, wait a minute or two and try again*

    Any issues, please contact jake.williams@marys.org.uk""")
    
    with st.sidebar:
        st.image(logo, width=75)
        st.header("About QLIPpy")
        st.markdown("""
        **QLIPpy** is your AI assistant for questions about QLIP (Quality, Leadership and Impact Partnership).
        
        *Please note: this is an experimental AI chatbot. The answers will be as accurate as possible, but there is room for error.*

        After talking to QLIPpy. Please get in touch with us to discuss further via our website: https://qlip.org.uk
        """)
        st.markdown("---")
        st.header("Example Questions")
        st.markdown("""
            - What is MEL?
            - How can I use the MEL framework?
            - What support does QLIP offer?
        """)
    
    query = st.text_area(
        "Enter your question",
        placeholder="Type your question here...",
        height=68
    )
    
    if st.button("Submit", type="primary"):
        if query:
            with st.spinner("Processing your query..."):
                process_query(query, False)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()