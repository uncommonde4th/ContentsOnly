from cx_Freeze import setup, Executable
import sys
import os

# Dependencies are automatically detected, but it might need fine tuning.
build_exe_options = {
    "packages": [
        "os", "sys", "tkinter", "cv2", "numpy", "PIL", 
        "logging", "pathlib", "dataclasses", "typing"
    ],
    "include_files": [
        ("resources/", "resources/"),
    ],
    "excludes": ["test", "unittest", "email", "http", "urllib", "xml"],
    "optimize": 2,
}

# GUI applications require a different base on Windows (the default is for a
# console application).
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="ContentsOnly",
    version="1.0.0",
    description="Автоматическая обрезка документов из фотографий",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "src/app.py",
            base=base,
            target_name="DocumentScanner.exe",
            icon="resources/app.ico"
        )
    ]
)
