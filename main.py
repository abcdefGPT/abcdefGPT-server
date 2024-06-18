from sqlalchemy.future import select
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from entity import AsyncSessionLocal, Temp, create_tables
from llm import prototype_rag  # llm.py 파일의 prototype_llm_rag 함수 임포트
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
import openai
from dotenv import load_dotenv

load_dotenv()
GPT_API_KEY = os.getenv("GPT_API_KEY")
os.environ["OPENAI_API_KEY"] = GPT_API_KEY
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await create_tables()
    await create_vector_store()

async def create_vector_store():

    loader = DirectoryLoader('rule/public_institution_rule_english', glob="*.json", show_progress=True,
                             loader_cls=TextLoader)
    docs = loader.load()
    vectorstore = FAISS.from_documents(documents=docs, embedding=OpenAIEmbeddings(api_key=openai.api_key))
    app.state.vectorstore = vectorstore


async def get_db():
    async with AsyncSessionLocal() as db:
        yield db


@app.get("/")
async def read_root(query: str):
    try:
        llm = ChatOpenAI(
            temperature=1.0,  # 창의성 (0.0 ~ 2.0)
            max_tokens=2048,  # 최대 토큰수
            model_name='gpt-4o',  # 모델명
        )

        answer = prototype_rag.run(query, app.state.vectorstore, llm)  # 벡터스토어를 함수로 전달
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/db-test")
async def read_temp(db: AsyncSession = Depends(get_db)):
    async with db:
        result = await db.execute(select(Temp))
        temps = result.scalars().all()
        return temps
