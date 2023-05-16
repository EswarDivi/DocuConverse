
# DocuConverse

## Conversational AI Chatbot - Interact with Documents like never before

`The Art of Building Intelligent Applications with Langchain and Streamlit`

This is a Streamlit app that allows users to chat with a PDF document using a conversational AI model. The app uses [Cohere](https://cohere.com/) for language modeling and question answering, and [Chroma](https://github.com/chroma-core/chroma) for document indexing and [Langchain](https://github.com/hwchase17/langchain) for chaining all these together.

## Want to See Langchain Colab Notebook

Use This Colab Notebook: **Click** [here](https://colab.research.google.com/drive/1ZrQzc1RLEH7m1v86rykdRFfHqJMqjumw?usp=sharing)
by [@log-xp](https://www.github.com/log-xp) and [@Nikhil-Paleti](https://github.com/Nikhil-Paleti)

## Demo

For PDF Chatbot

<https://chatwithpdf.streamlit.app/>

<https://huggingface.co/spaces/eswardivi/ChatwithPdf/>

For Widgets (Streamlit Demo)

<https://widgets.streamlit.app/>

## Usage

To use the app, follow these steps:

1. Upload a PDF document using the sidebar.
2. Type your message in the "You:" field and press "Send".
3. The AI model will generate a response based on the contents of the PDF document.
4. The response will be displayed in the chat window.

You can adjust the temperature of the AI model and the chunk size for splitting the document using the sliders in the sidebar.

## Run Locally

Clone the project

```bash
  git clone https://github.com/EswarDivi/Anokha_Demo
```

Go to the project directory

```bash
  cd Anokha_Demo
```

To use this app, you will need to create an account with [Cohere](https://cohere.ai/) and get an API key. Once you have an API key, create a file `secrets.toml` in the root directory of this project and add the following line:

```toml
cohere_apikey="<your_api_key>"
```

Install dependencies

```bash
    pip install -r requirements.txt
```

## Deployment

To deploy this project run

```bash
  streamlit run Talkwithpdf.py
```

## Deploying on Streamlit Cloud

To deploy this project on Streamlit Sharing, follow the steps below:

1. Create an account on [Streamlit Sharing](https://streamlit.io/sharing) and connect it to your GitHub account.

2. Fork this repository to your GitHub account.

3. In the app secrets of your Streamlit Sharing dashboard, add a new secret named `cohere_apikey` and set it to your Cohere API key.

4. Click on **Deploy** and wait for the deployment to finish.

5. Once the deployment is finished, you can access your app on the provided URL.

Note: Make sure your Cohere API key is kept secret and is not exposed to the public.

## Credits

This app was created using the following libraries:

- [Streamlit](https://streamlit.io/)
- [Cohere](https://cohere.ai/)
- [Chroma](https://github.com/Language-Chain/chroma)
- [Langchain](https://github.com/hwchase17/langchain)
