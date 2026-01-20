from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = None
    STT_MODEL: str = "whisper-1"
    SUMMARY_MODEL: str = "gemini-flash-latest"

    class Config:
        env_file = ".env"


settings = Settings()
