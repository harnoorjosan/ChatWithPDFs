import streamlit as st # for frontend
from dotenv import load_dotenv # for loading .env variables
from PyPDF2 import PdfReader # for getting raw text from PDFs
from langchain.text_splitter import CharacterTextSplitter # for getting chunks of texts from raw text
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
#from langchain.embeddings import 
from langchain_community.vectorstores import FAISS
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace

def get_raw_text(docs):
    raw_text = ""
    for doc in docs: # looping over each PDF
        pdf_reader = PdfReader(doc) # PdfReader can extract pages
        for page in pdf_reader.pages: # looping over each page
            raw_text += page.extract_text()
    return raw_text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

'''def get_convo_chain(vector_store):
    llm = ChatHuggingFace()
    memory = ConversationBufferMemory(memory_key="chat history", return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return convo_chain'''


def main():
    load_dotenv() # loading .env variables
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:") # ":xxxx:" is used for emojis
    st.header("Chat with PDFs :books:")
    st.text_input("Type your question here....")

    if "conversation" not in st.session_state:
        st.session_state.convo_chain =  None

    with st.sidebar:
        st.subheader("Your PDFs")
        docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)  # accept multiple docs
        button = st.button("Process")
        if button: # button pressed
            with st.spinner("Processing....."): # a spinner for processing
                raw_text = get_raw_text(docs) # extract raw text from pdf
                chunks = get_text_chunks(raw_text) # divide raw text into chunks
                vector_store = get_vector_store(chunks)# store in vector database
                #st.session_state will bind the convo to chat therefore will stay persistent. Streamlit otw loads the code whenever there is an action and the convo will lose
                #st.session_state.convo_chain = get_convo_chain(vector_store)# start a conversation chain


if __name__ == '__main__':
    main()
    
