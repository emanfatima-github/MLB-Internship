from sqlalchemy.orm import Session

from .models import Job


def create_job(db: Session, job: Job):
    db.add(job)
    db.commit()
    db.refresh(job)

    return job


def get_job(db: Session, job_id: str):
    return db.query(Job).filter(
        Job.job_id == job_id
    ).first()


def get_jobs(
    db: Session,
    status: str | None = None,
    user_id: int | None = None
):
    query = db.query(Job)

    if user_id is not None:
        query = query.filter(
            Job.user_id == user_id
        )

    if status:
        query = query.filter(
            Job.status == status
        )

    return query.order_by(
        Job.created_at.desc()
    ).all()


def update_job(
    db: Session,
    job_id: str,
    **kwargs
):
    job = get_job(db, job_id)

    if not job:
        return None

    for key, value in kwargs.items():
        setattr(job, key, value)

    db.commit()
    db.refresh(job)

    return job


def delete_job(db: Session, job_id: str):
    job = get_job(db, job_id)

    if not job:
        return False

    db.delete(job)
    db.commit()

    return True