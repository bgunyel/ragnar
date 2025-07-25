import os
from pydantic_settings import BaseSettings

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "RAGNAR: Retrieval Augmented Generation based kNowledge AdviseR"

    LLM_BASE_URL: str
    TAVILY_API_KEY: str
    VLLM_API_KEY: str
    GROQ_API_KEY: str
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    LANGSMITH_API_KEY: str
    LANGSMITH_TRACING: str
    SUPABASE_URL: str
    SUPABASE_SECRET_KEY: str

    OUT_FOLDER: str = os.path.join(ENV_FILE_DIR, 'out')

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')

settings = Settings()
