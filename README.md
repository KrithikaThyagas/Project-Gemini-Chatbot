# Project-Gemini-Chatbot
# Chat with PDF using Gemini

This project allows users to interact with PDF documents through a conversational AI model. The application extracts text from uploaded PDFs, processes it into a searchable format using Pinecone, and then uses a conversational AI model to answer questions based on the content of the PDFs.

## Features

- **PDF Upload**: Upload multiple PDF files for processing.
- **Text Extraction**: Extracts and processes text from PDFs.
- **Conversational AI**: Ask questions and get answers based on the content of the uploaded PDFs.
- **Pinecone Integration**: Uses Pinecone for efficient vector storage and retrieval.


Steps : 


1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Project-Gemini-Chatbot.git
   cd chat-pdf-gemini
   
2. Install the required packages:

Copy code
pip install -r requirements.txt


3. Set up environment variables:

Create a .env file in the root directory with the following content:

Copy code
PINECONE_API_KEY = "98730030-0cbf-4536-8fad-14be657ba60e"
GOOGLE_API_KEY = "AIzaSyBi68y5pC--khvZItyQBMOo2spFRzcRUz8"

4. Usage
Run the Streamlit app:
Copy code
streamlit run app.py
