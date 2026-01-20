import json
import asyncio

from google import genai
from google.genai import types

from app.config import settings
from app.schemas.summary import TopicSummaryRequest, SummaryResponse


_SYSTEM_INSTRUCTION = (
    "You summarize reading club data in warm, friendly Korean. Use a casual "
    "but polite tone (\"~요\") and write as if you are speaking to a member. "
    "Return strict JSON with keys: summary (string), highlights (array of strings), "
    "keywords (array of strings)."
)


def _configure_gemini() -> None:
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")


def _parse_summary_payload(content: str) -> SummaryResponse:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return SummaryResponse(summary=content)

    return SummaryResponse(
        summary=data.get("summary") or "",
        highlights=data.get("highlights"),
        keywords=data.get("keywords"),
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


async def summarize_stt(text: str) -> SummaryResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "You summarize meeting transcripts."
            ),
        },
        {"role": "user", "content": text},
    ]
    return await _summarize(messages)
