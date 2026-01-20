from fastapi import UploadFile


async def transcribe_audio(file: UploadFile) -> str:
    # TODO: STT model call (e.g., Whisper API)
    data = await file.read()
    _ = data
    # Placeholder
    return "transcribed text"
