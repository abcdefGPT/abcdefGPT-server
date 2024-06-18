from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

Base = declarative_base()

# 비동기적으로 테이블을 생성하는 함수
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class ChatGroup(Base):
    __tablename__ = "chatgroup"
    group_id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    chats = relationship("Chat", back_populates="group")

class Chat(Base):
    __tablename__ = "chat"
    group_id = Column(Integer, ForeignKey("chatgroup.group_id"), primary_key=True, index=True)
    question = Column(String(255), primary_key=True, index=True)  # Specified length for String
    answer = Column(String(255))  # Specified length for String

    group = relationship("ChatGroup", back_populates="chats")
