from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = None
    STT_MODEL: str = "whisper-1"
    SUMMARY_MODEL: str = "gemini-flash-latest"
    CLOVA_SPEECH_API_URL: Optional[str] = None
    CLOVA_CLIENT_ID: Optional[str] = None
    CLOVA_CLIENT_SECRET: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
