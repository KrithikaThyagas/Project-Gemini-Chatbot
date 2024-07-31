import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Configure Google API
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Initialize Pinecone client
def initialize_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API key is not set.")
        return None
    return Pinecone(api_key=api_key)

# Extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Create vector store from chunks
def get_vector_store(text_chunks, pc):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", task_type="retrieval_document")
    
    index_name = "qa"
    dimension = 768
    metric = 'cosine'

    # Check if index exists, create if not
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=dimension, 
            metric=metric,
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(index_name)
    vectorstore = LangchainPinecone.from_texts(text_chunks, embeddings, index_name=index_name)
    return vectorstore

# Get conversational chain
def get_conversational_chain(vectorstore):
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

    # Create the ConversationalRetrievalChain with the retriever
    retriever = vectorstore.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )

    return chain

# Handle user input and answer questions
def user_input(user_question, vectorstore):
    try:
        if not isinstance(user_question, str) or not user_question.strip():
            st.error("Please provide a valid question.")
            return

        st.write(f"Question: {user_question}")

        # Generate a response using the chain
        chain = get_conversational_chain(vectorstore)
        response = chain({
            "question": user_question
        }, return_only_outputs=True)

        # Display the response
        if "answer" in response:
            st.write("Reply: ", response["answer"])
        else:
            st.write("Sorry, I could not find an answer to your question.")

    except Exception as e:
        st.error(f"Error during chain execution: {e}")

def main():
    st.set_page_config(page_title="Chat PDF", page_icon=":books:")
    st.header("Chat with PDF using Gemini💁")

    # Initialize Pinecone client
    pc = initialize_pinecone()
    if pc is None:
        st.stop()

    # File uploader and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text extracted from the PDFs. Please check the files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        vectorstore = get_vector_store(text_chunks, pc)
                        st.session_state['vectorstore'] = vectorstore
                        st.success("Document is Processed")

    # Handle user questions
    if 'vectorstore' in st.session_state:
        user_question = st.text_input("Ask a Question from the PDF Files")
        if user_question:
            user_input(user_question, st.session_state['vectorstore'])
    else:
        st.warning("Please upload and process PDF files first.")

if __name__ == "__main__":
    main()
