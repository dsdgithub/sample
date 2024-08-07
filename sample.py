

#!pip install streamlit-chat
#!pip install streamlit
#!pip install langchain
#!pip install faiss-cpu


import streamlit as st
from streamlit_chat import message
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain

# 단계 1: 문서 로드(Load Documents)
pdf_path = "C:/Users/ksrim/wiset/data/★2024 서울교육 주요업무(확정본).pdf"
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()


# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)


# 단계 3: 임베딩(Embedding) 생성
embeddings = OpenAIEmbeddings()


# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)


# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()


# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Answer in Korean.

    #Question: 
    {question} 
    #Context: 
    {context} 

    #Answer:"""
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4", temperature=0)

# 단계 8: 체인(Chain) 생성
chain = RunnableSequence(
    {
        "context": retriever,
        "question": RunnablePassthrough()
    },
    prompt,
    llm,
    StrOutputParser()
)

# # Streamlit 앱 초기 설정
# st.title("서울시 교육청 챗봇")
# st.write("질문을 입력하면 챗봇이 답변합니다.")

# # 사용자 입력 처리
# question = st.text_input("질문을 입력하세요:")
# if question:
#     response = chain.invoke(question)
#     st.write("답변:", response)



chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-4'), retriever=retriever)




st.set_page_config(
    page_title="교육TALK", # 크롬창 상단에 보일 제목
    page_icon="🏫",    # 크롬창 상단에 보일 아이콘 모양 설정
    
)



# 기업 마크 이미지 경로 설정 (로컬 파일 또는 URL)
logo = "C:/Users/ksrim/wiset/data/seoultalk.png"  # 로컬 파일 경로

# 화면 상단에 기업 마크 이미지 삽입
st.image(logo, width=700)





def conversational_chat(query):  #문맥 유지를 위해 과거 대화 저장 이력에 대한 처리      
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))        
    return result["answer"]
    
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["서울교육 주요 업무에 관해 무엇이든 질문해주세요!🏫"]


if 'past' not in st.session_state:
    st.session_state['past'] = ["안녕하세요! 질문있습니다."]



        
#챗봇 이력에 대한 컨테이너
response_container = st.container()

#사용자가 입력한 문장에 대한 컨테이너
container = st.container()

with container: #대화 내용 저장(기억)
    with st.form(key='Conv_Question', clear_on_submit=True):           
        user_input = st.text_input("질문:", placeholder="2024년 서울교육 주요업무에 대해 질문해주세요 (:", key='input')
        submit_button = st.form_submit_button(label='Send')
            
    if submit_button and user_input:
        output = conversational_chat(user_input)
            
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style = "fun-emoji", seed = "Nala")
            message(st.session_state["generated"][i], key=str(i), avatar_style = "bottts", seed = "Fluffy")




# 하단에 텍스트 추가
footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 30px;  /* 화면 하단에서 20px 위로 올림 */
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 18px;  /* 글씨 크기 조정 */
    }
    </style>
    <div class="footer">
         문의전화 정보화담당관 ☎️ 02-0000-0000
    </div>
    """

st.markdown(footer, unsafe_allow_html=True)