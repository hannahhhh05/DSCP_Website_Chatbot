import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def load_vector_store(website_url):
    # Load document from the given URL
    doc_loader = WebBaseLoader(website_url)
    raw_document = doc_loader.load()
    
    # Split document into smaller parts
    splitter = RecursiveCharacterTextSplitter()
    document_parts = splitter.split_documents(raw_document)
    
    # Generate a vector store from document parts
    vector_store = Chroma.from_documents(document_parts, OpenAIEmbeddings())
    
    return vector_store

def create_retriever_chain(vector_store):
    language_model = ChatOpenAI()
    
    retriever = vector_store.as_retriever()
    
    chat_prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(language_model, retriever, chat_prompt)
    
    return retriever_chain

def create_chat_chain(retriever_chain):
    language_model = ChatOpenAI()
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions using the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ])
    
    doc_chain = create_stuff_documents_chain(language_model, chat_prompt)
    
    return create_retrieval_chain(retriever_chain, doc_chain)

def fetch_response(user_query):
    retriever_chain = create_retriever_chain(st.session_state.vector_store)
    chat_chain = create_chat_chain(retriever_chain)
    
    response = chat_chain.invoke({
        "history": st.session_state.history,
        "input": user_query
    })
    
    return response['answer']

# Streamlit application setup
st.set_page_config(page_title="Ecolume Website Chatbot", page_icon="🤖")
st.title("Website Chatbot")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    website_url = st.text_input("Enter the website URL")

if not website_url:
    st.info("Please provide a website URL")

else:
    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = [
            AIMessage(content="Hello, I'm your AI assistant, EcoLume. How can I help you today?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_vector_store(website_url)

    # Process user input
    user_query = st.chat_input("Enter your message here...")
    if user_query:
        response = fetch_response(user_query)
        st.session_state.history.append(HumanMessage(content=user_query))
        st.session_state.history.append(AIMessage(content=response))

    # Display the conversation
    for message in st.session_state.history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.write(message.content)