# src/ragnar/apps/fastapi_app.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import os
from typing import AsyncGenerator, Optional
import datetime
import logging

from config import settings
from ragnar import BusinessIntelligenceAgent, get_llm_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance - initialized during startup
bia: Optional[BusinessIntelligenceAgent] = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup
    global bia
    try:
        llm_config = get_llm_config()

        bia = BusinessIntelligenceAgent(
            llm_config=llm_config,
            web_search_api_key=settings.TAVILY_API_KEY,
            database_url=settings.SUPABASE_URL,
            database_key=settings.SUPABASE_SECRET_KEY
        )
        logger.info("RAGNAR Business Intelligence Agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        raise
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("RAGNAR API shutting down")

app = FastAPI(
    title="RAGNAR Business Intelligence API",
    description="AI-powered business intelligence assistant",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins = [str(origin).rstrip("/") for origin in settings.BACKEND_CORS_ORIGINS] + [settings.FRONTEND_HOST],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    token_usage: dict


@app.middleware("http")
async def log_requests(request, call_next):
    start_time = datetime.datetime.now()
    response = await call_next(request)
    process_time = (datetime.datetime.now() - start_time).total_seconds()

    logger.info(
        f"{request.method} {request.url} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.2f}s"
    )
    return response


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "RAGNAR API",
        "agent_ready": bia is not None,
        "timestamp": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    return {
        "status": "operational",
        "timestamp": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat(),
        "service": "RAGNAR Business Intelligence API"
    }


@app.post("/api/v1/chat")
async def chat_endpoint(chat_message: ChatMessage) -> ChatResponse:
    if bia is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # At this point, bia is guaranteed to be a BusinessIntelligenceAgent instance
    assert bia is not None  # Type assertion for static analysis
    try:
        # Use your existing run method
        result = bia.run(query=chat_message.message)
        return ChatResponse(
            response=result['content'],
            token_usage=result['token_usage']
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(chat_message: ChatMessage):
    if bia is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # At this point, bia is guaranteed to be a BusinessIntelligenceAgent instance
    assert bia is not None  # Type assertion for static analysis

    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Use your existing stream_response method!
            for chunk in bia.stream_response(user_message=chat_message.message):
                yield f"data: {json.dumps({'chunk': chunk, 'status': 'streaming'})}\n\n"
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'error': str(e), 'status': 'error'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/v1/status")
async def detailed_status():
    """Detailed status endpoint for monitoring"""
    try:
        # Test database connection using your existing method
        if bia is not None:
            _ = bia.fetch_company_by_name("Test Company")
            db_status = "connected"
            agent_status = "ready"
        else:
            db_status = "unknown"
            agent_status = "not_initialized"
    except Exception as e:
        db_status = f"error: {str(e)}"
        agent_status = "error"

    return {
        "service": "RAGNAR Business Intelligence API",
        "status": "operational" if bia is not None else "degraded",
        "timestamp": datetime.datetime.now().astimezone(settings.TIME_ZONE).isoformat(),
        "components": {
            "database": db_status,
            "agent": agent_status,
            "models": ["language_model", "reasoning_model"] if bia is not None else []
        }
    }


# For local development and testing
if __name__ == "__main__":
    import uvicorn

    # Setup environment variables (same as your Streamlit app)
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    uvicorn.run(app, host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=False)
