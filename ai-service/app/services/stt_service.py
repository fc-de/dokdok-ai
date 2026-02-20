import os
import logging
from typing import Optional

import json

import httpx

from app.config import settings
from app.schemas.stt import (
    KeyPointResponse,
    SttRequest,
    SttResponse,
    SttSummaryResponse,
    TopicSummaryResponse,
)
from app.services.summary_service import summarize_stt


_ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ac3",
    ".ogg",
    ".flac",
    ".avi",
    ".mp4",
    ".mov",
    ".wmv",
    ".flv",
    ".mkv",
}
_MAX_FILE_BYTES = 50 * 1024 * 1024
_logger = logging.getLogger(__name__)


def _validate_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    if not os.path.isfile(path):
        return "INVALID_FILE"
    _, ext = os.path.splitext(path)
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        return "UNSUPPORTED_FORMAT"
    size = os.path.getsize(path)
    if size > _MAX_FILE_BYTES:
        return "FILE_TOO_LARGE"
    return None


def _clova_headers() -> dict[str, str]:
    if not settings.CLOVA_CLIENT_SECRET:
        raise ValueError("CLOVA credentials are not configured")
    headers = {"X-CLOVASPEECH-API-KEY": settings.CLOVA_CLIENT_SECRET}
    if settings.CLOVA_CLIENT_ID:
        headers["X-CLOVASPEECH-CLIENT-ID"] = settings.CLOVA_CLIENT_ID
    return headers


async def _call_clova_stt(path: str, language: str) -> str:
    if not settings.CLOVA_SPEECH_API_URL:
        raise ValueError("CLOVA_SPEECH_API_URL is not configured")
    _logger.info("CLOVA STT request start: path=%s, language=%s", path, language)
    headers = _clova_headers()
    params = {
        "language": language,
        "completion": "sync",
        "fullText": True,
        "wordAlignment": True,
        "noiseFiltering": True,
    }
    data = {"params": json.dumps(params, ensure_ascii=False)}
    timeout = httpx.Timeout(60.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(path, "rb") as f:
            files = {"media": (os.path.basename(path), f, "application/octet-stream")}
            response = await client.post(
                settings.CLOVA_SPEECH_API_URL,
                data=data,
                files=files,
                headers=headers,
            )
    response.raise_for_status()
    payload = response.json()
    text = payload.get("text")
    if text:
        _logger.info("CLOVA STT response text length=%d", len(text))
        return text
    result = payload.get("result", "FAILED")
    message = payload.get("message", "CLOVA response missing text")
    raise ValueError(f"CLOVA result={result}, message={message}")


async def _transcribe_to_text(req: SttRequest) -> tuple[Optional[str], Optional[str]]:
    if not req.filePath:
        return "", None

    error = _validate_file(req.filePath)
    if error:
        _logger.warning("STT file validation failed: %s", error)
        return None, error

    try:
        text = await _call_clova_stt(req.filePath, req.language)
        return text, None
    except Exception as exc:
        _logger.exception("STT transcription failed")
        return None, str(exc)
    finally:
        try:
            if req.filePath:
                os.remove(req.filePath)
        except OSError:
            pass


async def transcribe_audio(req: SttRequest) -> SttResponse:
    text, error = await _transcribe_to_text(req)
    if error:
        return SttResponse(jobId=req.jobId, status="FAILED", errorMessage=error)
    return SttResponse(jobId=req.jobId, status="DONE", text=text or "")


async def transcribe_and_summarize(req: SttRequest) -> SttSummaryResponse:
    _logger.info(
        "STT summarize request: meetingId=%s, jobId=%s, preAnswers=%s",
        req.meetingId,
        req.jobId,
        len(req.preAnswers) if req.preAnswers else 0,
    )
    text, error = await _transcribe_to_text(req)
    if error:
        raise ValueError(error)

    summary = await summarize_stt(text or "", req.preAnswers)
    _logger.info(
        "Summary generated: summary_len=%d, main_points=%d",
        len(summary.summary or ""),
        len(summary.mainPoints or []),
    )
    key_points = [KeyPointResponse(title=point, details=[]) for point in summary.mainPoints or []]

    topics: list[TopicSummaryResponse] = []
    if req.preAnswers:
        seen: set[int] = set()
        for item in req.preAnswers:
            if item.topicId is None or item.topicId in seen:
                continue
            seen.add(item.topicId)
            topics.append(
                TopicSummaryResponse(
                    topicId=item.topicId,
                    topicTitle=item.topicTitle,
                    summary=summary.summary,
                    keyPoints=key_points or None,
                )
            )
    if not topics:
        topics.append(
            TopicSummaryResponse(
                topicId=req.topicId,
                confirmOrder=req.confirmOrder,
                topicTitle=req.topicTitle,
                topicDescription=req.topicDescription,
                summary=summary.summary,
                keyPoints=key_points or None,
            )
        )

    meeting_id = req.meetingId if req.meetingId is not None else req.jobId
    return SttSummaryResponse(meetingId=meeting_id, topics=topics)
