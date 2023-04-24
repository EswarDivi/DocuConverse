import openai
import streamlit as st
from streamlit_chat import message
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
import base64
from langchain.embeddings import CohereEmbeddings
import cohere
from langchain.prompts import PromptTemplate

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


if not os.path.exists("./Temp_Files"):
    os.makedirs("./Temp_Files")
st.set_page_config(page_title="Chat With PDF", page_icon=":smile:")

tab1, tab2 = st.tabs(["📈 Chat", "🗃 PDF"])

tab1.markdown(
    "<h1 style='text-align: center;'>Chat With PDF</h1>",
    unsafe_allow_html=True,
)


def save_uploadedfile(uploadedfile):
    with open(
        os.path.join("Temp_Files", uploadedfile.name),
        "wb",
    ) as f:
        f.write(uploadedfile.getbuffer())
    return st.sidebar.success("Saved File:{} to tempDir".format(uploadedfile.name))


with st.sidebar:
    st.title("Upload PDF")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
    temperature = st.slider("Temperature", 0.1, 0.9, 0.5, 0.1)
    max_tokens = st.slider("Max Tokens", 16, 512, 192, 16)
    counter_placeholder = st.empty()
    clear_button = st.button("Clear Conversation", key="clear")

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=5)

flan_ul2 = HuggingFaceHub(
    repo_id="google/flan-ul2",
    model_kwargs={
        "temperature": temperature,
        "max_new_tokens": max_tokens,
    },
)

embeddings = CohereEmbeddings(
    model="large", cohere_api_key="4ReeiiO3StqicxM8vJT4cWiIL51jT7MLMWOkQdRw"
)


if uploaded_file is not None:
    save_uploadedfile(uploaded_file)
    tab1.markdown(
        "<h3 style='text-align: center;'>Now You Are Chatting With "
        + uploaded_file.name
        + "</h3>",
        unsafe_allow_html=True,
    )


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    tab2.markdown(pdf_display, unsafe_allow_html=True)


def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """Use the following pieces of context to answer the question at the end.
    You are A chatbot designed for helping with PDF files, specifically for communicating with them. 
    If you are unable to provide an answer to a question, simply state that you do not know rather than attempting to provide a false or inaccurate response.

    {context}

    Question: {question}"""
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
        llm=flan_ul2,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"


def generate_response(query):
    result = qa({"query": query})
    print(result["result"])
    print(result["source_documents"])
    return result["result"]


if uploaded_file is not None:
    PDF_loader("Temp_Files/" + uploaded_file.name)

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]


response_container = tab1.container()
container = tab1.container()

with container:
    with st.form(key="my_form", clear_on_submit=True):
        user_input = st.text_area("You:", key="input", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        output = generate_response(user_input)
        print(output)
        st.session_state["past"].append(user_input)
        st.session_state["generated"].append(output)


if st.session_state["generated"]:
    with response_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# show pdf in tab2

if uploaded_file is not None:
    show_pdf("Temp_Files/" + uploaded_file.name)
