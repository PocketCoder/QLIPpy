import os
import time
import streamlit as st
from groq import Groq
from typing import List, Dict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from PIL import Image

EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = Groq()

st.set_page_config(page_title="QLIPpy", page_icon="🤖", layout="centered")

logo_path = "static/logo.png"
logo = Image.open(logo_path)


def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0


def add_message(role: str, content: str):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.messages.append(
        {"role": role, "content": content, "timestamp": timestamp}
    )
    st.session_state.message_count += 1

    # Keep only the last 10 messages
    if st.session_state.message_count > 10:
        st.session_state.messages.pop(0)
        st.session_state.message_count -= 1


def get_groq_response(system_prompt: str, user_prompt: str) -> str:
    try:
        full_response = ""
        start_time = time.time()

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            stream=True,
            temperature=0.5,
            max_completion_tokens=5024,
            top_p=1,
            stop=None,
        )

        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content

        elapsed_time = time.time() - start_time
        st.caption(f"Response generated in {elapsed_time:.2f} seconds")

        return full_response

    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return f"Error: {str(e)}"


@st.cache_resource
def setup_retriever():
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

        db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
        )
        return db.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Error setting up retriever: {e}")
        raise


def process_query(query: str) -> str:
    if not query.strip():
        return "Please enter a query."

    try:
        retriever = setup_retriever()
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

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
           - Use **bold** for emphasis or to mark sections of your response
           - Use `code` for specific terms or references
           - Use > for important quotes or highlights
           - Use bullet points (*) and new lines for lists
           - Format links properly: [text](url)
           - Make sure to add new lines so that text chunks are easier to read.
        7. Always output contact information:
        - For queries regarding training, supervision, and first aid, contact Sally Baxter, Mary's CEO & Youth Development Manager, at sally.baxter@marys.org.uk.
        - For queries regarding the MEL Framework, and bespoke support, contact Ruth Virgo, London Youth's Youth Sector Development Manager, at ruth.virgo@londonyouth.org.
        - We also send out a monthly newsletter to keep you and the sector up to date with what we do.
          Subscribe here: https://mailchi.mp/ba929b010b40/qlip-newsletter
        """

        user_prompt = f"""Based on this context:
        {context}
        
        Please answer this question:
        {query}"""

        return get_groq_response(system_prompt, user_prompt)

    except Exception as e:
        st.error(f"Error processing query: {e}")
        return f"Error processing query: {e}"


def display_chat_messages():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    message["content"],
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant", avatar=logo):
                st.markdown(message["content"], unsafe_allow_html=True)


def main():
    initialize_chat_history()

    st.title("QLIPpy")
    st.markdown("Ask questions about QLIP.")
    st.markdown(
        """
    *Your first question will take a few seconds to load, but after that it should be quick!*

    *We have a limit of 30 questions a minute. If it doesn't work, wait a minute or two and try again*

    Any issues, please contact jake.williams@marys.org.uk"""
    )

    with st.sidebar:
        st.image(logo, width=75)
        st.header("About QLIPpy")
        st.markdown(
            """
        **QLIPpy** is your AI assistant for questions about QLIP (Quality, Leadership and Impact Partnership).
        
        *Please note: this is an experimental AI chatbot. The answers will be as accurate as possible, but there is room for error.*

        After talking to QLIPpy. Please get in touch with us to discuss further via our website: https://qlip.org.uk
        """
        )
        st.markdown("---")
        st.header("Example Questions")
        st.markdown(
            """
            - What is MEL?
            - How can I use the MEL framework?
            - What support does QLIP offer?
        """
        )

        # Add clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.message_count = 0
            st.rerun()

    # Chat container
    chat_container = st.container()

    with chat_container:
        display_chat_messages()

    # Chat input
    if query := st.chat_input(placeholder="Type your question here..."):
        add_message("user", query)
        with st.spinner("Processing your query..."):
            response = process_query(query)
            add_message("assistant", response)
        st.rerun()


if __name__ == "__main__":
    main()
