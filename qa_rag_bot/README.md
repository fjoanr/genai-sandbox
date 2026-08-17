# Practice Exercise – Multi-PDF RAG Question-Answering Bot

## Task

Build a RAG-powered question-answering bot in Python that can handle multiple PDF files instead of just one. The bot should allow the user to upload several PDFs through a Streamlit interface and then ask questions across all of them. For each answer, also display the source file names where the information was retrieved from.

## Suggested Steps

1. Streamlit UI for file upload

    - Create a Streamlit app (app.py).
    - Add a file uploader to allow multiple PDFs to be uploaded.
    - Extract the text from all uploaded PDFs.

2. Chunk and embed

    - Split the text into smaller chunks.
    - Generate embeddings for each chunk and store them in a vector store (in-memory or supported DB).

3. Retriever

    - Search across all PDFs to find the most relevant chunks for a given query.
    - Keep track of the file name each chunk came from.

4. Query with LLM

    - Pass the retrieved chunks and the user’s question into the LLM.
    - Generate a grounded, context-based answer.

5. Display results in Streamlit

    - Show the chatbot’s answer in the app.
    - Below the answer, display the source file names that were used to generate it.

## What you will practice

- Applying the RAG pipeline with multiple documents
- Using Streamlit for UI to upload files and interact with the chatbot
- Building a multi-document knowledge assistant
- Displaying sources alongside answers for reliability

👉 Upload 2–3 PDFs and ask questions that might require information from more than one file. Check how the app grounds its answers and shows which documents were used.
