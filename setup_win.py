from cx_Freeze import setup, Executable
import sys
import os

# Добавляем src в PYTHONPATH
sys.path.insert(0, 'src')

build_exe_options = {
    "packages": ["os", "sys", "tkinter", "cv2", "numpy", "PIL"],
    "include_files": [
        ("resources/", "resources/"),
    ],
    "excludes": ["test", "unittest"],
    "optimize": 2,
}

setup(
    name="ContentsOnly",
    version="1.0.0",
    description="Document Scanner",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "src/app.py",
            base="Win32GUI",
            target_name="ContentsOnly.exe"
        )
    ]
)
