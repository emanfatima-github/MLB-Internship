import os
import uuid

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

MAX_FILE_SIZE = 100 * 1024 * 1024  

ALLOWED_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}


def create_directories():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def generate_job_id():
    return f"job_{uuid.uuid4().hex[:10]}"


def generate_request_id():
    return f"req_{uuid.uuid4().hex[:10]}"


def get_file_extension(filename: str):
    return os.path.splitext(filename)[1].lower()


def is_allowed_file(filename: str):
    extension = get_file_extension(filename)
    return extension in ALLOWED_EXTENSIONS