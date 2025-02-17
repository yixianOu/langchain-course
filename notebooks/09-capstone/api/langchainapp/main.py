import asyncio

from fastapi import APIRouter, FastAPI
from fastapi.responses import StreamingResponse

from .routers import agent_exec

app = FastAPI()
# initilizing our application

# Router setup
chat_router = APIRouter(prefix='/chat', tags=['chat'])

@chat_router.get("/query")
async def query(query: str):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = agent_exec.QueueCallbackHandler(queue)
    return StreamingResponse(agent_exec.token_generator(query, streamer), media_type="text/plain")

app.include_router(router=chat_router)
# including our routers (in this case only one), to the application