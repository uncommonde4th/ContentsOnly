from cx_Freeze import setup, Executable
import sys
import os

# Добавляем src в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Создаем папку resources, если её нет
if not os.path.exists("resources"):
    os.makedirs("resources")
    print("Создана папка resources (пустая)")

# Проверяем наличие файлов в resources
if not any(os.scandir("resources")):
    print("ВНИМАНИЕ: Папка resources пуста!")

build_exe_options = {
    "packages": [
        "os", "sys", "tkinter", "cv2", "numpy", "PIL", 
        "logging", "pathlib", "dataclasses", "typing",
        "scanner", "gui", "utils"
    ],
    "include_files": [
        # Если папка resources существует и не пуста
        ("resources", "resources"),
    ],
    "excludes": ["test", "unittest", "email", "http", "urllib", "xml"],
    "optimize": 2,
    "include_msvcr": False,
    "silent": False,
}

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
            target_name="ContentsOnly",
            icon=None  # Добавьте путь к иконке если есть
        )
    ]
)