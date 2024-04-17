__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_cohere import ChatCohere
from langchain_cohere import CohereRerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.chains import (
    RetrievalQA,
    ConversationalRetrievalChain,
    StuffDocumentsChain,
    LLMChain,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Chat With PDF", page_icon=":smile:")

if not os.path.exists("./tempfolder"):
    os.makedirs("./tempfolder")

tab1, tab2, tab3 = st.tabs(
    ["💬 Chat with PDF", "📚 Relevant Document Chunks", "💾 Current Database in Memory"]
)

tab1.markdown(
    """
    <h1 style='text-align: center;'>Chat With PDF</h1>
    """,
    unsafe_allow_html=True,
)


def save_uploadedfile(uploadedfile):
    with open(
        os.path.join("tempfolder", uploadedfile.name),
        "wb",
    ) as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved File")


with st.sidebar:
    st.markdown(
        "<h2 style='text-align: center; color: #007BFF;'>Upload PDF</h2>",
        unsafe_allow_html=True,
    )
    cohere_api_key = st.text_input("Enter your Cohere API key", type="password")
    if cohere_api_key:
        try:
            test_model = ChatCohere(model="command", cohere_api_key=cohere_api_key)
            response = test_model.invoke("Hello")
            if "error" in response:
                raise Exception("Invalid API key")
        except Exception as e:
            st.error(f"Error: {str(e)} - Please enter a correct Cohere API key.")
            st.stop()
    else:
        st.error("Please enter your Cohere API key to proceed.")
    with st.expander("Contact Information"):
        st.markdown("For any queries, please feel free to contact:")
        st.markdown(
            "Email: [eswar.divi.902@gmail.com](mailto:eswar.divi.902@gmail.com)"
        )
        st.markdown("GitHub: [github.com/EswarDivi](https://github.com/EswarDivi)")

    with st.expander("Additional Information"):
        st.info("Get Your API key at https://dashboard.cohere.com/api-keys")
        st.markdown(
            "<h4 style='text-align: center;'>Powered by Cohere</h4>",
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

    with st.expander("Adjust Settings"):
        temp_r = st.slider("Temperature", 0.1, 0.9, 0.45, 0.1)
        chunksize = st.slider("Chunk Size for Splitting Document", 256, 1024, 400, 10)
        clear_button = st.button("Clear Conversation", key="clear")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50, separators=[" ", ",", "\n"]
)

if cohere_api_key:
    embeddings = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key,
    )
    model_cohere = ChatCohere(
        model="command",
        cohere_api_key=cohere_api_key,
    )


def PDF_loader(document):
    db_path = f"./tempfolder/db_{os.path.basename(document).split('.')[0]}"
    if os.path.exists(db_path):
        print("Using Cached One")
        db_chroma = Chroma(embedding_function=embeddings, persist_directory=db_path)
        retriever = db_chroma.as_retriever()
    else:
        loader = PyPDFLoader(document)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)
        db_chroma = Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=db_path,
        )
        retriever = db_chroma.as_retriever()

    compressor = CohereRerank(cohere_api_key=cohere_api_key)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

    prompt_template = """You are an AI chatbot that helps users chat with PDF documents.
    Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you find the answer, write the answer in a Elegant way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.
    Example:
    The Answer is derived from[1] this page
    [1] Source_ Page:PageNumber

    {context}

    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nPageNumber:{page}\nsource:{source}",
    )
    llm_chain = LLMChain(
        llm=model_cohere, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
        verbose=False,
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        callbacks=None,
        verbose=False,
        retriever=compression_retriever,
        return_source_documents=True,
    )

    return qa


if uploaded_file is not None and cohere_api_key is not None:
    save_uploadedfile(uploaded_file)
    file_size = os.path.getsize(f"tempfolder/{uploaded_file.name}") / (1024 * 1024)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Uploaded PDF: {file_size} MB")
    qa = PDF_loader("tempfolder/" + uploaded_file.name)
    tab1.markdown(
        "<h3 style='text-align: center;'>Now You Are Chatting With "
        + uploaded_file.name
        + "</h3>",
        unsafe_allow_html=True,
    )


def generate_response(query, qa):
    result = qa.invoke({"query": query, "chat_history": ""})

    tab2.markdown(
        "<h3 style='text-align: center;'>Relevant Documents Metadata</h3>",
        unsafe_allow_html=True,
    )

    tab2.write(result["source_documents"])
    result["result"] = result["result"]
    return result["result"]


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if uploaded_file is not None and cohere_api_key is not None:
        data = {"question": prompt}
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            while not full_response:
                with st.spinner("Thinking..."):
                    Output = generate_response(prompt, qa)
                    full_response = Output if Output else "Failed to get the response."
                fr = ""
                full_response = str(full_response)
                for i in full_response:
                    import time

                    time.sleep(0.02)
                    fr += i
                    message_placeholder.write(fr + "▌")
                message_placeholder.write(f"{full_response}")
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.write(
                "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered"
            )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Please go ahead and upload the PDF in the sidebar, it would be great to have it there and make sure API key Entered",
            }
        )

if clear_button:
    st.session_state.messages = []

with tab3:
    files = os.listdir("./tempfolder")
    file_data = [
        {
            "File Name": file,
            "Size (MB)": os.path.getsize(f"./tempfolder/{file}") / (1024 * 1024),
        }
        for file in files
    ]
    st.write("### Current Database Files in Memory")
    st.table(file_data)
