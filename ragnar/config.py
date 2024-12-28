import os
from pydantic_settings import BaseSettings

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "Ragnar"

    BACKEND_PORT: int = 8080
    HOST: str = "0.0.0.0"
    ENABLE_RELOAD: bool = False

    UI_PORT: int = 8501
    INPUT_FOLDER: str
    OUT_FOLDER: str
    MODEL: str

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')


settings = Settings()
