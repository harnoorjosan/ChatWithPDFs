import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

def get_raw_text(docs):
    raw_text = ""
    for doc in docs:
        pdf_reader = PdfReader(doc) # PdfReader can extract pages
        for page in pdf_reader.pages: # looping over each page
            raw_text += page.extract_text()
    return raw_text



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:") # ":xxxx:" is used for emojis
    st.header("Chat with PDFs :books:")
    st.text_input("Type your question here....")
    with st.sidebar:
        st.subheader("Your PDFs")
        docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)  # accept multiple docs
        button = st.button("Process")
        if button: # button pressed
            with st.spinner("Processing....."): # a spinner for processing
                # extract raw text from pdf
                raw_text = get_raw_text(docs)
                st.write(raw_text)
                
                # divide raw text into chunks
                
                # store in vector database
if __name__ == '__main__':
    main()
    
