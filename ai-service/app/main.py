from fastapi import FastAPI
from app.api import stt, summary

app = FastAPI(title="Dokdok AI Service")

app.include_router(stt.router, prefix="/stt", tags=["stt"])
app.include_router(summary.router, prefix="/summaries", tags=["summary"])
