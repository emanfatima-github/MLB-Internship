from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./database.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_database():
    inspector = inspect(engine)

    if "jobs" in inspector.get_table_names():
        columns = [
            column["name"]
            for column in inspector.get_columns("jobs")
        ]

        if "user_id" not in columns:
            with engine.begin() as connection:
                connection.execute(
                    text("ALTER TABLE jobs ADD COLUMN user_id INTEGER")
                )