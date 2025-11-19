import os
import datetime
from pydantic_settings import BaseSettings
from pydantic import SecretStr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "RAGNAR: Retrieval AuGmented kNowledge AdviseR"

    TIME_ZONE: datetime.timezone = datetime.timezone(offset=datetime.timedelta(hours=3), name='UTC+3')

    ANTHROPIC_API_KEY: SecretStr = ""
    GROQ_API_KEY: SecretStr = ""
    LANGSMITH_API_KEY: SecretStr = ""
    LANGSMITH_TRACING: str = "false"
    OLLAMA_API_KEY: SecretStr = ""
    OPENAI_API_KEY: SecretStr = ""
    SUPABASE_URL: SecretStr = ""
    SUPABASE_SECRET_KEY: SecretStr = ""
    TAVILY_API_KEY: SecretStr = ""

    OUT_FOLDER: str = os.path.join(ENV_FILE_DIR, 'out')

    BACKEND_PORT: int = 8080
    BACKEND_HOST: str = "0.0.0.0"

    FRONTEND_HOST: str = "http://localhost:5173"
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:8000"]

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')

settings = Settings()
