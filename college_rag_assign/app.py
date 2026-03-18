import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion import load_uploaded_files

st.set_page_config(page_title="College Student Knowledge Assistant", page_icon="🎓", layout="wide")

st.title("🎓 College Student Knowledge Assistant")
st.write("Ask questions about college policies, schedules, courses, and more. Answers are based on uploaded documents.")

# Upload documents
st.subheader("Upload Knowledge Documents")

uploaded_files = st.file_uploader(
    "Upload PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:

    if "vectorstore" not in st.session_state:

        with st.spinner("Processing uploaded documents..."):

            docs = load_uploaded_files(uploaded_files)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            embeddings = OllamaEmbeddings(model="nomic-embed-text")

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.session_state.vectorstore = vectorstore

            st.success(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")


question = st.text_input(
    "Enter your question",
    placeholder="Example: What is the minimum attendance required?"
)

if st.button("Get Answer"):

    if "vectorstore" not in st.session_state:
        st.warning("Please upload documents first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:

        vectorstore = st.session_state.vectorstore

        with st.spinner("Retrieving and generating answer..."):

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

            retrieved_docs = retriever.invoke(question)

            if not retrieved_docs:
                st.error("No relevant information found in the documents.")

            else:

                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

                llm = ChatOllama(model="phi3", temperature=0)

                prompt = f"""
Answer only from the context below.
If the answer is not available in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

Question:
{question}
"""

                response = llm.invoke(prompt)

                st.subheader("Answer")
                st.write(response.content)

                st.subheader("Sources")

                for i, doc in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**Source {i}: {doc.metadata.get('source','Unknown')}**")
                    st.write(doc.page_content)
                    st.divider()