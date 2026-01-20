from fastapi import APIRouter
from app.schemas.summary import (
    TopicSummaryRequest, SttSummaryRequest, SummaryResponse
)
from app.services.summary_service import summarize_preanswers, summarize_stt

router = APIRouter()


@router.post("/topics", response_model=SummaryResponse)
async def preanswers_summary(req: TopicSummaryRequest):
    return await summarize_preanswers(req)


@router.post("/stt", response_model=SummaryResponse)
async def stt_summary(req: SttSummaryRequest):
    return await summarize_stt(req.text)
