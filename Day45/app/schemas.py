from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    total_detections: int
    output_file: Optional[str] = None

    class Config:
        from_attributes = True


class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str