from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="user", nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)

    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
        index=True
    )

    filename = Column(String, nullable=False)

    status = Column(
        String,
        nullable=False,
        default="queued"
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    completed_at = Column(
        DateTime,
        nullable=True
    )

    processing_time = Column(
        Float,
        nullable=True
    )

    total_detections = Column(
        Integer,
        default=0
    )

    output_file = Column(
        String,
        nullable=True
    )