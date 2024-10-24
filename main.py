from sqlalchemy.future import select
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from entity import AsyncSessionLocal, ChatGroup, Chat, create_tables
import rag
from dto import ChatRequest
# Import the LLM configuration from the separate file
from llm.llm_config import get_entities, get_re, convert_rener, query_decomposition, retriever, rag_chain

app = FastAPI()

@app.on_event("startup")
async def startup():
    await create_tables()
    # await create_vector_store(app)

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db

# 채팅 그룹 삭제
@app.delete('/delete-chat-group')
async def delete_chat_group(group_id: int, db: AsyncSession = Depends(get_db)):
    try:
        async with db.begin():
            result = await db.execute(select(ChatGroup).where(ChatGroup.group_id == group_id))
            chat_group = result.scalar_one_or_none()
            if chat_group is None:
                raise HTTPException(status_code=404, detail="존재하지 않는 채팅 그룹입니다.")

            await db.delete(chat_group)
            await db.commit()

        return {"detail": "채팅 그룹이 삭제되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 모든 채팅 그룹 반환
@app.get('/all-chat-group')
async def all_chat_group(db: AsyncSession = Depends(get_db)):

    try:
        async with db.begin():
            result = await db.execute(select(ChatGroup))
            chat_groups = result.scalars().all()

        return chat_groups
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 특정 채팅 그룹의 모든 질문과 답변 반환
@app.get('/all-chat')
async def all_chat(group_id: int, db: AsyncSession = Depends(get_db)):
    try:
        async with db.begin():
            result = await db.execute(select(Chat).where(Chat.group_id == group_id))
            chats = result.scalars().all()

        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat')
async def chat(chatRequest: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        query = chatRequest.query

        decom_question = []

        entities = get_entities(query)
        relations = get_re(entities, query)
        rener_json = convert_rener(entities, relations)

        decom_query = query_decomposition(rener_json)
        decom_question.extend([decom_query])

        answers = []
        contexts = []
        query_and_contexts = []

        for queries in decom_question:
            temp_contexts = []
            for query in queries:
                t_c = []
                for docs in retriever.get_relevant_documents(query):
                    t_c.append(docs.page_content)
                    temp_contexts.append(docs.page_content)
                query_and_contexts.append({'query': query, 'document': t_c})
            contexts.append(temp_contexts)
            answers.append(rag_chain.invoke({"context": temp_contexts, "question": query}))

        # answer = rag.user_chat(chatRequest.query, app.state.vectorstore, llm, llm_chain)  # 벡터스토어를 함수로 전달
        async with db.begin():
            if chatRequest.group_id == -1:
                # 새로운 채팅 그룹 생성
                new_group = ChatGroup()
                db.add(new_group)
                await db.flush()  # 새로운 그룹의 ID를 얻기 위해 flush 호출
                final_group_id = new_group.group_id
            else:
                # 기존 채팅 그룹 사용
                result = await db.execute(select(ChatGroup).where(ChatGroup.group_id == chatRequest.group_id))
                existing_group = result.scalar_one_or_none()
                if existing_group is None:
                    raise HTTPException(status_code=404, detail="존재하지 않는 채팅 그룹입니다.")
                final_group_id = existing_group.group_id

            # 채팅 저장
            new_chat = Chat(group_id=final_group_id, question=chatRequest.query, answer=answers[0])
            db.add(new_chat)
            await db.commit()

        return {"answer": answers, "group_id": final_group_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import asyncio
    import uvicorn

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, loop="asyncio", http="h11")