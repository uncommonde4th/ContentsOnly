from cx_Freeze import setup, Executable
import sys
import os

# Добавляем src в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

build_exe_options = {
    "packages": [
        "os", "sys", "tkinter", "cv2", "numpy", "PIL", 
        "logging", "pathlib", "dataclasses", "typing",
        "scanner", "gui", "utils"  # ЯВНО добавляем ваши пакеты
    ],
    "include_files": [
        ("resources/", "resources/"),
    ],
    "excludes": ["test", "unittest", "email", "http", "urllib", "xml"],
    "optimize": 2,
}

base = None

setup(
    name="ContentsOnly",
    version="1.0.0",
    description="Автоматическая обрезка документов из фотографий",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "src/app.py",
            base=base,
            target_name="ContentsOnly"
        )
    ]
)