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
from app.services.summary_service import summarize_stt, summarize_topic


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

    # 토픽 목록 추출 (중복 제거, 순서 유지)
    topic_map: dict[int, dict] = {}
    if req.preAnswers:
        for item in req.preAnswers:
            if item.topicId is not None and item.topicId not in topic_map:
                topic_map[item.topicId] = {
                    "topicId": item.topicId,
                    "topicTitle": item.topicTitle or "",
                    "topicDescription": item.topicDescription or "",
                    "confirmOrder": item.confirmOrder,
                }

    topics: list[TopicSummaryResponse] = []

    if topic_map:
        # 토픽별로 개별 요약 생성
        for idx, (topic_id, topic_info) in enumerate(topic_map.items(), start=1):
            _logger.info("Summarizing topic: topicId=%s, title=%s", topic_id, topic_info["topicTitle"])

            result = await summarize_topic(
                topic_id=topic_id,
                topic_title=topic_info["topicTitle"],
                topic_description=topic_info["topicDescription"],
                transcript=text or "",
                preanswers=req.preAnswers or [],
            )

            key_points = [
                KeyPointResponse(
                    title=kp.get("title", ""),
                    details=kp.get("details", []),
                )
                for kp in result.get("keyPoints", [])
            ]

            confirm_order = topic_info["confirmOrder"] if topic_info["confirmOrder"] else idx

            topics.append(
                TopicSummaryResponse(
                    topicId=topic_id,
                    confirmOrder=confirm_order,
                    topicTitle=topic_info["topicTitle"],
                    topicDescription=topic_info["topicDescription"] or None,
                    summary=result.get("summary", ""),
                    keyPoints=key_points or None,
                )
            )

            _logger.info(
                "Topic summary generated: topicId=%s, summary_len=%d, keyPoints=%d",
                topic_id,
                len(result.get("summary", "")),
                len(key_points),
            )
    else:
        # preAnswers가 없는 경우 기존 방식 (단일 토픽)
        summary = await summarize_stt(text or "", req.preAnswers)
        key_points = [KeyPointResponse(title=point, details=[]) for point in summary.mainPoints or []]

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
