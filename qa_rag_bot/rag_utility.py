import os

import pypdf
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

# Load the model from OpenAI
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# llm = ChatOllama(
#    model="gemma3:270m",
#    temperature=0.0
# )


def process_document_to_chroma_db(file_name):
    # Load the PDF using PyPDF (recommended by LangChain)
    reader = pypdf.PdfReader(f"{working_dir}/{file_name}")

    # process each page in the PDF and create a Document from them
    pages = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_name, "page": i + 1},
        )
        for i, page in enumerate(reader.pages)
    ]

    # Split the text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=False,
    )
    texts = text_splitter.split_documents(pages)

    # Store the document chunks in a Chroma vector database
    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore",
    )

    return 0


def answer_question(user_question):
    # Load the persistent Chroma vector database
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore", embedding_function=embedding
    )
    # Create a retriever for document search
    retriever = vectordb.as_retriever()

    # Create a RetrievalQA chain to answer user questions using Llama-3.3-70B
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain.invoke({"query": user_question})
