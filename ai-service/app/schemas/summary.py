from typing import List, Optional

from pydantic import BaseModel


class TopicAnswerInput(BaseModel):
    userId: int
    content: str


class TopicSummaryRequest(BaseModel):
    topicId: int
    topicTitle: str
    answers: List[TopicAnswerInput]


class SttSummaryRequest(BaseModel):
    text: str


class SummaryResponse(BaseModel):
    summary: str
    highlights: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
