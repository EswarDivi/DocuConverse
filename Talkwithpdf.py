# Import Required Libraries
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import os
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere
from datetime import datetime

# Setting Up Streamlit Page
st.set_page_config(page_title="Chat With PDF", page_icon=":smile:", layout="wide")

# Creating Temp Folder
if not os.path.exists("./tempfolder"):
    os.makedirs("./tempfolder")

# Styling
st.markdown(
    """
    <style>
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .main {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        .chat-message {
            background-color: #f1f1f1;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .chat-input {
            border: none;
            border-radius: 10px;
            padding: 10px;
            background-color: #f1f1f1;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# tabs
tab1, tab2 = st.tabs(["📈 Chat Here", "🗃 Relevant Chunks"])

with tab1:
    st.markdown(
        """
        <div class="main">
            <h1 style='text-align: center; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;'>Chat With PDF</h1>
            <h4 style='text-align: center; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;'>Powered by Cohere</h4>
            <p style='text-align: center;'>For uninterrupted usage, visit the <a href='https://huggingface.co/spaces/eswardivi/ChatwithPdf' target='_blank'>HuggingFace Space</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Saving Upload file to tempfolder
def save_uploadedfile(uploadedfile):
    with open(
        os.path.join("tempfolder", uploadedfile.name),
        "wb",
    ) as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved File")

# Creating Sidebar for Utilites
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-content">
            <h2 style='font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;'>Upload PDF</h2>
            <p>For any Queries, please feel free to contact:</p>
            <p><a href="mailto:eswar.divi.902@gmail.com">Email</a></p>
            <p><a href="https://github.com/EswarDivi" target="_blank">GitHub</a></p>
            <hr>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    temp_r = st.slider("Temperature", 0.1, 0.9, 0.45, 0.1)
    chunksize = st.slider("Chunk Size for Splitting Document ", 256, 1024, 400, 10)
    clear_button = st.button("Clear Conversation", key="clear")

# Initialzing Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunksize, chunk_overlap=10, separators=[" ", ",", "\n"])

# Intializing Cohere Embdedding
embeddings = CohereEmbeddings(model="large", cohere_api_key=st.secrets["cohere_apikey"])

def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """
    System Prompt:
    Your are an AI chatbot that helps users chat with PDF documents. How may I help you today?

    {context}

    {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    texts = text_splitter.split_documents(documents)

    # Create a new instance of Chroma with a unique persist_directory for each file
    db = Chroma.from_documents(texts,embeddings,persist_directory=f"./tempfolder/db_{os.path.basename(document).split('.')[0]}")
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(
        llm=Cohere(
            model="command-xlarge-nightly",
            temperature=temp_r,
            cohere_api_key=st.secrets["cohere_apikey"],
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return qa

if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    file_size = os.path.getsize(f"tempfolder/{uploaded_file.name}") / (1024 * 1024)  # Size in MB
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Uploaded PDF: {file_size} MB")
    qa = PDF_loader("tempfolder/" + uploaded_file.name)
    tab1.markdown(
        """
        <div class="main">
            <h3 style='text-align: center; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;'>Now You Are Chatting With {}</h3>
        </div>
        """.format(uploaded_file.name),
        unsafe_allow_html=True,
    )

def generate_response(query, qa):
    result = qa({"query": query, "chat_history": ""})

    tab2.markdown(
        """
        <div class="main">
            <h3 style='text-align: center; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;'>Relevant Documents Metadata</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab2.write(result["source_documents"])
    result["result"] = result["result"]
    return result["result"]

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message("message-container", message["role"]):
        st.markdown(f"<div class='chat-message'>{message['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("What is up?", key="chat_input", placeholder="Type your message here...", disabled=False):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-message'>{prompt}</div>", unsafe_allow_html=True)
    if uploaded_file is not None:
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
                    message_placeholder.markdown(f"<div class='chat-message'>{fr}▌</div>", unsafe_allow_html=True)
                message_placeholder.markdown(f"<div class='chat-message'>{full_response}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("<div class='chat-message'>Please go ahead and upload the PDF in the sidebar, it would be great to have it there.</div>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": "Please go ahead and upload the PDF in the sidebar, it would be great to have it there."})

# Enabling Clear button
if clear_button:
    st.session_state.messages = []