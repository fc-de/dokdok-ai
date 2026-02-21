from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.schemas.api_response import ApiResponse
from app.schemas.stt import SttRequest
from app.services.stt_service import transcribe_audio, transcribe_and_summarize

router = APIRouter()


@router.post("", response_model=ApiResponse)
async def stt(req: SttRequest):
    try:
        data = await transcribe_audio(req)
        payload_data = data.model_dump()
        if data.status == "FAILED":
            payload = ApiResponse(
                code="FAILED",
                message=data.errorMessage or "STT failed",
                data=payload_data,
            )
            return JSONResponse(status_code=400, content=payload.model_dump())
        return ApiResponse(code="SUCCESS", message="요청이 성공했습니다.", data=payload_data)
    except Exception as exc:
        payload = ApiResponse(code="FAILED", message=str(exc), data=None)
        return JSONResponse(status_code=400, content=payload.model_dump())


@router.post("/summary", response_model=ApiResponse)
async def stt_summary(req: SttRequest):
    try:
        data = await transcribe_and_summarize(req)
        return ApiResponse(
            code="SUCCESS",
            message="AI 요약 조회 성공",
            data=data.model_dump(),
        )
    except Exception as exc:
        payload = ApiResponse(code="FAILED", message=str(exc), data=None)
        return JSONResponse(status_code=400, content=payload.model_dump())
