from typing import Optional

from pydantic import BaseModel


class PreAnswerInput(BaseModel):
    topicId: Optional[int] = None
    topicTitle: Optional[str] = None
    userId: int
    content: str


class SttRequest(BaseModel):
    jobId: int
    filePath: Optional[str] = None
    language: str = "ko-KR"
    meetingId: Optional[int] = None
    topicId: Optional[int] = None
    confirmOrder: Optional[int] = None
    topicTitle: Optional[str] = None
    topicDescription: Optional[str] = None
    preAnswers: Optional[list[PreAnswerInput]] = None


class SttResponse(BaseModel):
    jobId: int
    status: str
    text: Optional[str] = None
    errorMessage: Optional[str] = None


class KeyPointResponse(BaseModel):
    title: str
    details: list[str]


class TopicSummaryResponse(BaseModel):
    topicId: Optional[int] = None
    confirmOrder: Optional[int] = None
    topicTitle: Optional[str] = None
    topicDescription: Optional[str] = None
    summary: Optional[str] = None
    keyPoints: Optional[list[KeyPointResponse]] = None


class SttSummaryResponse(BaseModel):
    meetingId: int
    topics: list[TopicSummaryResponse]
