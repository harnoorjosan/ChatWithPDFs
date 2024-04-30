import streamlit as st
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:") # ":xxxx:" is used for emojis
    st.header("Chat with PDFs :books:")
    st.text_input("Type your question here....")
    with st.sidebar:
        st.subheader("Your PDFs")
        st.file_uploader("Upload PDFs here")
        st.button("Process")

if __name__ == '__main__':
    main()
    
