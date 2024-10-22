from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    group_id: int