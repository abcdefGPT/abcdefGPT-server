from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from entity import AsyncSessionLocal, Temp, create_tables
from sqlalchemy.future import select

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await create_tables()

async def get_db():
    async with AsyncSessionLocal() as db:
        yield db

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/temp/")
async def read_temp(db: AsyncSession = Depends(get_db)):
    async with db:
        result = await db.execute(select(Temp))
        temps = result.scalars().all()
        return temps