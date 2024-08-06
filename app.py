import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_chatbot_template import css, bot_template, user_template
import os

groq_api_key = os.getenv('GROQ_API_KEY')


def extract_text(pdf_files):
    """
    Function to extract the text from a PDF file

    Args:
        pdf_file (file): The PDF files to extract the text from

    Returns:
        text (str): The extracted text from the PDF file
    """

    # Initialize the raw text variable
    text = ""

    # Iterate over the documents
    for pdf_file in pdf_files:
        print("[INFO] Extracting text from {}".format(pdf_file.name))

        # Read the PDF file
        pdf_reader = PdfReader(pdf_file)

        # Extract the text from the PDF pages and add it to the raw text variable
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_chunks(text):
    """
    Function to get the chunks of text from the raw text

    Args:
        text (str): The raw text from the PDF file

    Returns:
        chunks (list): The list of chunks of text
    """

    # Initialize the text splitter
    splitter = CharacterTextSplitter(
        separator="\n", # Split the text by new line
        chunk_size=10000, # Split the text into chunks of 1000 characters
        chunk_overlap=2000, # Overlap the chunks by 200 characters
        length_function=len # Use the length function to get the length of the text
    )

    # Get the chunks of text
    chunks = splitter.split_text(text)

    return chunks

def get_vectorstore(chunks):
    """
    Function to create avector store for the chunks of text to store the embeddings

    Args:
        chunks (list): The list of chunks of text

    Returns:
        vector_store (FAISS): The vector store for the chunks of text
    """

    # Initialize the embeddings model to get the embeddings for the chunks of text
    #embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',model_kwargs={'device':"cuda"})
    embeddings = HuggingFaceEmbeddings()

    # Create a vector store for the chunks of text embeddings
    # Can use any other online vector store (Elasticsearch, PineCone, etc.)
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store

def get_conversation_chain(vector_store,model_choice):
    """
    Function to create a conversation chain for the chat model

    Args:
        vector_store (FAISS): The vector store for the chunks of text
    
    Returns:
        conversation_chain (ConversationRetrievalChain): The conversation chain for the chat model
    """
    
    # Initialize the chat model using Langchain OpenAi API
    # Set the temperature and select the model to use
    # Can replace this with any other chat model (Llama, Falcom, etc.)
    #llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.1)
    #llm  = HuggingFaceHub(repo_id="google/flan-t5-xxl",model_kwargs={"temperature":0.9,"max_length":64}, client_options={"timeout": 180})
    #llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1",model_kwargs={"temperature": 0.9, "max_length": 256})
    # Initialize the chat memory
    #llm = HuggingFacePipeline.from_model_id(model_id="google/flan-t5-xl",task="text2text-generation",max_length=512,temperature=0.7)
    if model_choice == 'Llama3-8b-8192':
        llm = ChatGroq(model_name='Llama3-8b-8192')
        model_name = 'Llama3-8b-8192'
    if model_choice == 'Mixtral-8x7b-32768':
        llm = ChatGroq(model_name='mixtral-8x7b-32768')
        model_name = 'Mixtral-8x7b-32768'
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create a conversation chain for the chat model
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, # Chat model
        retriever=vector_store.as_retriever(), # Vector store
        memory=memory, # Chat memory
    )

    return conversation_chain,model_name

def generate_response(question):
    """
    Function to generate a response for the user query using the chat model

    Args:
        question (str): The user query

    Returns:
        response (str): The response from the chat model
    """

    # Get the response from the chat model for the user query
    response = st.session_state.conversations({'question': question})

    # Update the chat history
    st.session_state.chat_history = response['chat_history']

    # Add the response to the UI
    for i, message in enumerate(st.session_state.chat_history):
        # Check if the message is from the user or the chatbot
        if i % 2 == 0:
            # User message
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            # Chatbot message
            st.write(bot_template.replace(
                "{{MSG}}", message.content).replace(
                "{{MODEL}}",st.session_state.model_name), unsafe_allow_html=True)


## Landing page UI
def run_UI():
    """
    The main UI function to display the UI for the webapp

    Args:
        None

    Returns:
        None
    """

    # Load the environment variables (API keys)
    #load_dotenv()

    # Set the page tab title
    st.set_page_config(page_title="DocuMentor", page_icon="🤖", layout="wide")

    # Add the custom CSS to the UI
    st.write(css, unsafe_allow_html=True)

    # Initialize the session state variables to store the conversations and chat history
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Set the page title
    st.header("DocuMentor: Conversations with Your Data 🤖")

    # Input text box for user query
    user_question = st.text_input("Upload your data and ask me anything?")

    # Check if the user has entered a query/prompt
    if user_question:
        # Call the function to generate the response
        generate_response(user_question)

    # Sidebar menu
    with st.sidebar:
        st.subheader("Document Uploader")

        # Document uploader
        pdf_files = st.file_uploader("Upload a document you want to chat with", type="pdf", key="upload", accept_multiple_files=True)
        
        model_choice = st.selectbox("Choose a model",["Llama3-8b-8192","Mixtral-8x7b-32768"],key="model_choice")

        # Process the document after the user clicks the button
        if st.button("Start Chatting ✨"):
            # Add a progress spinner
            with st.spinner("Processing"):
                # Convert the PDF to raw text
                raw_text = extract_text(pdf_files)
                
                # Get the chunks of text
                chunks = get_chunks(raw_text)
                
                # Create a vector store for the chunks of text
                vector_store = get_vectorstore(chunks)

                # Create a conversation chain for the chat model
                st.session_state.conversations,st.session_state.model_name = get_conversation_chain(vector_store,model_choice)

# Application entry point
if __name__ == "__main__":
    # Run the UI
    run_UI()
