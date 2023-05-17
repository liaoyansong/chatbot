import openai
from langchain.chat_models import ChatOpenAI
from llama_index import Document, SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, \
    QuestionAnswerPrompt
# from langchain import OpenAI
import os
from llama_index import PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from llama_index import LangchainEmbedding, ServiceContext
from llama_index import StorageContext, load_index_from_storage
import streamlit as st
from streamlit_chat import message
import requests
# import simplejson as json

key = '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'
os.environ['OPENAI_API_KEY'] = key

# LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
# 文档划分
max_input_size = 4096
num_output = 256
prompt_helper = PromptHelper(max_input_size, num_output, 20)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
# 生成index
# documents = SimpleDirectoryReader('./data').load_data()
# index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
# 保存index
# index.storage_context.persist(persist_dir="./index")

# 读取index
storage_context = StorageContext.from_defaults(persist_dir="./index")
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
index = load_index_from_storage(
    storage_context=storage_context,
    service_context=service_context
)

# 问答
QA_PROMPT_TMPL = (
    "我们提供了一些背景知识\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "如果可以请基于这些知识回答问题: {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)
query_engine = index.as_query_engine(text_qa_template=QA_PROMPT)

#---------------------------------------------------------------------------------------
def generate_response(prompt):
    response = str(query_engine.query(prompt))
    return response

st.title("眼底疾病诊断系统")

# 不显示聊天记录
# prompt = st.text_input("You:")
# response = generate_response(prompt)
# st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True)


# 显示聊天记录
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input = st.text_input("You:", key='input')
if user_input:
    st.session_state['past'].append(user_input)
    bot_output = generate_response(user_input)
    st.session_state['generated'].append(bot_output)
    
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['past'][i], is_user=True)
        message(st.session_state["generated"][i], is_user=False)
