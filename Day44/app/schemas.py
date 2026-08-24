from pydantic import BaseModel
from datetime import datetime
from typing import Optional


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
