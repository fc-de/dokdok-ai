from typing import Any, Optional

from pydantic import BaseModel


class ApiResponse(BaseModel):
    code: str
    message: str
    data: Optional[Any] = None
