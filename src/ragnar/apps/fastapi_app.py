# src/ragnar/apps/fastapi_app.py
import datetime
import logging
import os
from typing import Optional, Any

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from ragnar import BusinessIntelligenceAgent, get_llm_config, DatabaseTable

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
    content: str
    token_usage: dict
    cost_list: list[dict[str, Any]]
    total_cost: float


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
        result = await bia.run(query=chat_message.message)
        return ChatResponse(
            content=result['content'],
            token_usage=result['token_usage'],
            cost_list=result['cost_list'],
            total_cost=result['total_cost'],
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status")
async def detailed_status():
    """Detailed status endpoint for monitoring"""
    try:
        # Test database connection using your existing method
        if bia is not None:
            _ = bia.list_all_names(table_name=DatabaseTable.COMPANIES)
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
            "models": bia.get_model_names() if bia is not None else []
        }
    }


# For local development and testing
if __name__ == "__main__":
    import uvicorn

    # Setup environment variables (same as your Streamlit app)
    os.environ['LANGSMITH_API_KEY'] = getattr(settings, 'LANGSMITH_API_KEY', '')
    os.environ['LANGSMITH_TRACING'] = getattr(settings, 'LANGSMITH_TRACING', 'false')

    uvicorn.run(app, host=settings.BACKEND_HOST, port=settings.BACKEND_PORT, reload=False)
