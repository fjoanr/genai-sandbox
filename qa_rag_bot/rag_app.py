import os

import streamlit as st
from rag_utility import answer_question, process_document_to_chroma_db

# set the working directory
working_dir = os.path.dirname(os.path.abspath((__file__)))

# streamlit page setup
st.set_page_config(
    page_title="QA-RAG",
    page_icon="🤖",
    layout="centered",
)

st.title("💎 QA RAG Bot 💎")
st.text(
    """\
Do you need help understanding large PDF files? \
Share them with me, and then let me know what kind of information you require from them!"
"""
)

# file uploader widget
uploaded_files = st.file_uploader(
    "Upload your PDF file(s)", type=["pdf"], accept_multiple_files=True
)

for uploaded_file in uploaded_files:
    # define save path
    save_path = os.path.join(working_dir, uploaded_file.name)
    #  save the file
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chroma_db(uploaded_file.name)

    # remove the files from repository after being added to DB
    os.remove(save_path)

st.info("Documents Processed Successfully")


# text widget to get user input
user_question = st.text_area("What do you want to know from the given data?")

if st.button("Answer"):
    response = answer_question(user_question)

    st.markdown("========== Output ==========")
    st.markdown(response["result"])
    st.markdown("========== References ==========")
    for source in response["source_documents"]:
        doc_name = source.metadata["source"]
        page_idx = source.metadata["page"]
        st.markdown(f"📄 {doc_name}, page {page_idx}")
