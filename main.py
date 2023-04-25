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
from langchain.llms import Cohere
from langchain.memory import ConversationBufferWindowMemory


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


if not os.path.exists("./Temp_Files"):
    os.makedirs("./Temp_Files")
st.set_page_config(page_title="Chat With PDF", page_icon=":smile:")

tab1, tab2 = st.tabs(["📈 Chat Here", "🗃 PDF"])

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
    temp_r = st.slider("Temperature", 0.1, 0.9, 0.5, 0.1)
    max_tokens = st.slider("Max Tokens", 16, 512, 192, 16)
    counter_placeholder = st.empty()
    clear_button = st.button("Clear Conversation", key="clear")

text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=10)

flan_ul2 = HuggingFaceHub(
    repo_id="google/flan-ul2",
    model_kwargs={
        "temperature": temp_r,
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


tab2.header("Preview of PDF")


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'
    tab2.markdown(pdf_display, unsafe_allow_html=True)


def PDF_loader(document):
    loader = OnlinePDFLoader(document)
    documents = loader.load()
    prompt_template = """Use the following pieces of context to answer the question at the end.
    {context}
   I am a AI assistant was designed by Eswar Divi to assist users with PDF documents while ensuring their safety and efficiency. Assistant's top priority is the well-being of humans, and Assistant will not engage in any activity that could pose a threat. If Assistant is unable to provide a suitable response it will respond with unable to answer 
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
            cohere_api_key="4ReeiiO3StqicxM8vJT4cWiIL51jT7MLMWOkQdRw",
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs,
    )
    return "Ready"


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


def generate_response(query):
    result = qa({"query": query, "chat_history": st.session_state["chat_history"]})
    print(result)
    print(st.session_state["chat_history"])
    # print(result["source_documents"])
    result["result"] = result["result"]
    return result["result"].replace("The answer is ","")


if uploaded_file is not None:
    PDF_loader("Temp_Files/" + uploaded_file.name)

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []

response_container = tab1.container()
container = tab1.container()

with container:
    # with st.form(key="my_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="input")
    # submit_button = st.form_submit_button(label="Send")

    if user_input:
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


if clear_button:
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["chat_history"] = []
# show pdf in tab2

if uploaded_file is not None:
    show_pdf("Temp_Files/" + uploaded_file.name)
