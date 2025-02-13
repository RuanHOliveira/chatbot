import torch
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceHub

load_dotenv()

st.set_page_config(page_title="Virtual Assistant", page_icon="🤖")
st.title("Virtual Assistant")


def model_response(user_query, chat_history):
    llm = HuggingFaceHub(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={
            "temperature": 0.1,
            "return_full_text": False,
            "max_new_tokens": 512,
        },
    )

    system_prompt = """
    Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}.
    """
    language = "português"
    user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", user_prompt),
        ]
    )

    chain = prompt_template | llm | StrOutputParser()

    return chain.stream(
        {
            "chat_history": chat_history,
            "input": user_query,
            "language": language,
        }
    )


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou o seu assistente vistual! Como posso te ajudar?")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        resp = st.write_stream(
            model_response(user_query, st.session_state.chat_history)
        )

    st.session_state.chat_history.append(AIMessage(content=resp))
