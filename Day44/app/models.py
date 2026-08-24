from sqlalchemy import Column, String, DateTime, Float, Integer
from datetime import datetime

from .database import Base


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)

    filename = Column(String, nullable=False)

    status = Column(String, nullable=False, default="queued")

    created_at = Column(DateTime, default=datetime.utcnow)

    completed_at = Column(DateTime, nullable=True)

    processing_time = Column(Float, nullable=True)

    total_detections = Column(Integer, default=0)

    output_file = Column(String, nullable=True)
