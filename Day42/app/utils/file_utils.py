from pathlib import Path

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_allowed_video(filename: str) -> bool:
    extension = Path(filename).suffix.lower()
    return extension in ALLOWED_EXTENSIONS
