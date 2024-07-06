from sqlalchemy.future import select
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from entity import AsyncSessionLocal, ChatGroup, Chat, create_tables
import rag
from dto import ChatRequest
# Import the LLM configuration from the separate file
from llm.llm_config import llm, llm_chain, create_vector_store

app = FastAPI()

@app.on_event("startup")
async def startup():
    await create_tables()
    await create_vector_store(app)

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db

# 특정 채팅 그룹 삭제
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

@app.get('/all-chat-group')
async def all_chat_group():
    return 0

@app.get('/all-chat')
async def all_chat():
    return 0

@app.post('/chat')
async def chat(chatRequest: ChatRequest, db: AsyncSession = Depends(get_db)):
    try:
        answer = rag.user_chat(chatRequest.query, app.state.vectorstore, llm, llm_chain)  # 벡터스토어를 함수로 전달
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
            new_chat = Chat(group_id=final_group_id, question=chatRequest.query, answer=answer)
            db.add(new_chat)
            await db.commit()

        return {"answer": answer, "group_id": final_group_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import asyncio
    import uvicorn

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, loop="asyncio", http="h11")