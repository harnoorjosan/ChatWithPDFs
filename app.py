import streamlit as st # for frontend
from dotenv import load_dotenv # for loading .env variables
from PyPDF2 import PdfReader # for getting raw text from PDFs
from langchain.text_splitter import CharacterTextSplitter # for getting chunks of texts from raw text
#from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
#from langchain.embeddings import 
from langchain_community.vectorstores import FAISS
import torch
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.llms.huggingface_hub import HuggingFaceHub
from htmlTemplate import css, bot_template, user_template

def get_raw_text(docs):
    raw_text = ""
    # looping over each PDF
    for doc in docs: 
        # PdfReader can extract pages
        pdf_reader = PdfReader(doc) 
        # looping over each page
        for page in pdf_reader.pages: 
            raw_text += page.extract_text()
    return raw_text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def get_convo_chain(vector_store):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_lenght":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    convo_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return convo_chain

def handle_question(question):
    response = st.session_state.convo_chain({'question': question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0: # odd numbers of the history ie what the question is
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # loading .env variables
    load_dotenv() 
    # ":xxxx:" is used for emojis
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:") 
    st.write(css, unsafe_allow_html=True)
    st.header("Chat with PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.convo_chain =  None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history =  None

    question = st.text_input("Type your question here....")
    if question:
        handle_question(question)


    with st.sidebar:
        st.subheader("Your PDFs")
        # accept multiple docs
        docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)  
        button = st.button("Process")
        # button pressed
        if button: 
            # a spinner for processing
            with st.spinner("Processing....."): 
                 # extract raw text from pdf
                raw_text = get_raw_text(docs)
                # divide raw text into chunks
                chunks = get_text_chunks(raw_text) 
                # store in vector database
                vector_store = get_vector_store(chunks)
                #st.session_state will bind the convo to chat therefore will stay persistent. Streamlit otw loads the code whenever there is an action and the convo will lose
                st.session_state.convo_chain = get_convo_chain(vector_store)  # start a conversation chain


if __name__ == '__main__':
    main()
    
