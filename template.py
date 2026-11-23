
from pathlib import Path


# =========================
# PROJECT ROOT
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent


# =========================
# FOLDERS
# =========================

folders = [
    "data",
    "notebooks",
    "models",
    "templates",
    "static",
]


# =========================
# FILES
# =========================

files = [
    "app.py",
    "requirements.txt",
    ".gitignore",
    "README.md",
    "templates/index.html",
    "static/style.css",
]


# =========================
# CREATE FOLDERS
# =========================

for folder in folders:
    folder_path = PROJECT_ROOT / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"[CREATED/READY] Folder: {folder}")


# =========================
# CREATE FILES
# =========================

for file in files:
    file_path = PROJECT_ROOT / file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.touch(exist_ok=True)
    print(f"[CREATED/READY] File: {file}")


# =========================
# COMPLETE
# =========================

print("\nProject structure created successfully!")
