import json
import asyncio
from typing import Optional

from google import genai
from google.genai import types

from app.config import settings
from app.schemas.summary import TopicSummaryRequest, SummaryResponse
from app.schemas.stt import PreAnswerInput


_SYSTEM_INSTRUCTION = (
    "You summarize reading club discussions in Korean. Use a casual but polite "
    "tone (\"~요\"), but avoid chatty phrases and avoid thanking. "
    "Return strict JSON with keys: summary (string), mainPoints (array of strings). "
    "The summary string must follow this format:\n"
    "1) Title line: use the topic title if provided; otherwise infer a short title.\n"
    "2) Subtitle line: one short sentence describing the topic.\n"
    "3) A line with '핵심 요약' then a short paragraph (2-4 sentences).\n"
    "Do NOT include '주요 포인트' in the summary. Put points into mainPoints only.\n"
    "mainPoints should have 2-4 concise items."
)


def _configure_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")


def _parse_summary_payload(content: str) -> SummaryResponse:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return SummaryResponse(summary=content)

    if isinstance(data, list):
        if all(isinstance(item, str) for item in data):
            return SummaryResponse(summary="", mainPoints=data)
        if all(isinstance(item, dict) for item in data):
            summaries: list[str] = []
            points: list[str] = []
            for item in data:
                summary = item.get("summary")
                if isinstance(summary, str) and summary.strip():
                    summaries.append(summary.strip())
                main_points = item.get("mainPoints") or item.get("highlights")
                if isinstance(main_points, list):
                    points.extend([p for p in main_points if isinstance(p, str)])
            if summaries or points:
                return SummaryResponse(
                    summary="\n\n".join(summaries),
                    mainPoints=points or None,
                )
        return SummaryResponse(summary=content)

    if not isinstance(data, dict):
        return SummaryResponse(summary=content)

    return SummaryResponse(
        summary=data.get("summary") or "",
        mainPoints=data.get("mainPoints") or data.get("highlights"),
    )


def _extract_prompt(messages: list[dict[str, str]]) -> str:
    # Gemini SDK expects a single prompt; keep roles as simple headers.
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            continue
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts).strip()


def _summarize_sync(messages: list[dict[str, str]]) -> SummaryResponse:
    _configure_gemini()
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    prompt = _extract_prompt(messages)
    response = client.models.generate_content(
        model=settings.SUMMARY_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
        ),
    )
    content = response.text or "{}"
    return _parse_summary_payload(content)

async def _summarize(messages: list[dict[str, str]]) -> SummaryResponse:
    return await asyncio.to_thread(_summarize_sync, messages)


async def summarize_preanswers(req: TopicSummaryRequest) -> SummaryResponse:
    user_lines = "\n".join(
        f"- userId={item.userId}: {item.content}" for item in req.answers
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize reading club pre-answers for a topic."
            ),
        },
        {
            "role": "user",
            "content": (
                f"topicId={req.topicId}, topicTitle={req.topicTitle}\n"
                "Answers:\n"
                f"{user_lines}"
            ),
        },
    ]
    return await _summarize(messages)


def _format_preanswers(preanswers: list[PreAnswerInput]) -> str:
    lines: list[str] = []
    for item in preanswers:
        content = (item.content or "").strip()
        if not content:
            continue
        topic_part = f"topicId={item.topicId}"
        if item.topicTitle:
            topic_part += f", topicTitle={item.topicTitle}"
        lines.append(f"- {topic_part}, userId={item.userId}: {content}")
    return "\n".join(lines)


async def summarize_stt(
    text: str,
    preanswers: Optional[list[PreAnswerInput]] = None,
) -> SummaryResponse:
    cleaned_text = (text or "").strip()
    formatted = _format_preanswers(preanswers or [])
    if not cleaned_text and not formatted:
        raise ValueError("No content to summarize")

    user_content = cleaned_text
    if preanswers:
        if formatted:
            user_content = (
                "Transcript:\n"
                f"{cleaned_text}\n\n"
                "Pre-answers:\n"
                f"{formatted}"
            )
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize meeting transcripts."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return await _summarize(messages)
