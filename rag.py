import os
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()


# 디버깅을 위한 프로젝트명을 기입
os.environ["RAG_TEST"] = os.getenv("RAG_TEST")

# langsmith에서 trace 하기 위한 용도 + 우리 API key 설정
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["OPENAI_API_KEY"] = os.getenv("GPT_API_KEY")
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")

question = "안정적으로 장기 투자를 하고 싶습니다. 장기 투자가 유망한 종목을 알려주세요."
print("Q: " + question)

# PDF
# PDF 파일 업로드, 청크 나누고 인덱싱(PDF)
# from langchain.document_loaders import PyPDFLoader

# PDF 파일 로드. 파일의 경로 입력
loader = PyPDFLoader(PDF_FILE_PATH)

# 페이지 별 문서 로드
docs = loader.load_and_split()
# print(f"문서의 수: {len(docs)}")

# # 크롤링
# # 크롤링을 원하는 링크를 로드하고, 청크로 나누고 인덱싱
# loader = WebBaseLoader(
#     web_paths=("https://n.news.naver.com/article/437/0000378416",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             "div",
#             attrs={"class": ["newsct_article _article_body",
#                              "media_end_head_title"]},
#         )
#     ),
# )

# docs = loader.load()
# print(f"문서의 수: {len(docs)}")
# docs

# JSON
# from langchain_community.document_loaders import JSONLoader

# import json
# from pathlib import Path
# from pprint import pprint


# file_path = "/content/dataset/test.json"
# data = json.loads(Path(file_path).read_text())

# pprint(data)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

splits = text_splitter.split_documents(docs)
# print(splits)
len(splits)



# OpenAI embedding object 생성
openai_embeddings = OpenAIEmbeddings(api_key=openai.api_key)

# FAISS vectorstore 생성
vectorstore = FAISS.from_documents(documents=splits, embedding=openai_embeddings)

# document의 정보를 검색하고 생성.
retriever = vectorstore.as_retriever()




prompt = hub.pull("rlm/rag-prompt")

class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)


llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0,
    streaming=True,
    callbacks=[StreamCallback()],
)


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합치기
    return "\n\n".join(doc.page_content for doc in docs)


# 체인 생성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("A: ", end="")
print(rag_chain.invoke(question))  # 문서에 대한 질문 및 답변 출력