from fastapi import APIRouter, UploadFile, File
from app.schemas.stt import SttResponse
from app.services.stt_service import transcribe_audio

router = APIRouter()


@router.post("", response_model=SttResponse)
async def stt(file: UploadFile = File(...)):
    text = await transcribe_audio(file)
    return SttResponse(text=text)
