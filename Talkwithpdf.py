# Import Required Libraries
import os
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Cohere
import time
import threading

# Deleting the tempfolder every 2 hours
def delete_tempfolder():
    while True:
        # Delete the tempfolder directory
        if os.path.exists("./tempfolder"):
            # Remove all files and subdirectories in tempfolder
            for root, dirs, files in os.walk("./tempfolder", topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir("./tempfolder")
        
        # Sleep for 2 hours
        time.sleep(2 * 60 * 60)

# Start the function in a separate thread
thread = threading.Thread(target=delete_tempfolder)
thread.start()

# Setting Up API Tokens
# Create .streamlit Folder in Root Directory
# Create a File secrets.toml
# TOML format
# cohere_apikey="Enter you Key"


# Setting Up Streamlit Page
st.set_page_config(page_title="Chat With PDF", page_icon=":smile:")


# Creating Temp Folder
if not os.path.exists("./tempfolder"):
    os.makedirs("./tempfolder")


# tabs
tab1, tab2 = st.tabs(["📈 Chat Here", "🗃 Relevant Documents"])

tab1.markdown(
    "<h1 style='text-align: center;'>Chat With PDF</h1><h4 style='text-align: center;'>Powered by Cohere</h4><p style='text-align: center;'>For uninterrupted usage, visit the <a href='https://huggingface.co/spaces/eswardivi/ChatwithPdf' target='_blank'>Hugging Face model</a></p>",
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
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    temp_r = st.slider("Temperature", 0.1, 0.9, 0.3, 0.1)
    chunksize = st.slider("Chunk Size for Splitting Document ", 256, 1024, 300, 10)
    clear_button = st.button("Clear Conversation", key="clear")

# Initialzing Text Splitter
text_splitter = CharacterTextSplitter(chunk_size=chunksize, chunk_overlap=10)

# Intializing Cohere Embdedding
embeddings = CohereEmbeddings(model="large", cohere_api_key=st.secrets["cohere_apikey"])


def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """ 
    Your are an AI Chatbot devolped to help users to Chat to a PDF document.Use the following pieces of context to answer the question at the end.Greet Users!!
    {context}

    {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    texts = text_splitter.split_documents(documents)
    global db
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    global qa
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
    return "Ready"


if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    file_size = os.path.getsize(f"tempfolder/{uploaded_file.name}") / (1024 * 1024)  # Size in MB
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Uploaded PDF: {file_size} MB")
    PDF_loader("tempfolder/" + uploaded_file.name)
    tab1.markdown(
        "<h3 style='text-align: center;'>Now You Are Chatting With "
        + uploaded_file.name
        + "</h3>",
        unsafe_allow_html=True,
    )


# Session State
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []


# Generating Response
def generate_response(query):
    result = qa({"query": query, "chat_history": st.session_state["chat_history"]})

    tab2.markdown(
        "<h3 style='text-align: center;'>Relevant Documents Metadata</h3>",
        unsafe_allow_html=True,
    )

    tab2.write(result["source_documents"])
    result["result"] = result["result"]
    return result["result"]


# Creating Containers

response_container = tab1.container()
container = tab1.container()


with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_input("You:", key="input")
        submit_button = st.form_submit_button(label="Send")

    if user_input and submit_button:
        if uploaded_file is not None:
            output = generate_response(user_input)
            print(output)
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(output)
            st.session_state["chat_history"] = [(user_input, output)]
        else:
            st.session_state["past"].append(user_input)
            st.session_state["generated"].append(
                "Please go ahead and upload the PDF in the sidebar, it would be great to have it there."
            )

if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(
                st.session_state["past"][i],
                is_user=True,
                key=str(i) + "_user",
                avatar_style="adventurer",
                seed=123,
            )
            message(st.session_state["generated"][i], key=str(i))

# Enabling Clear button

if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["chat_history"] = []