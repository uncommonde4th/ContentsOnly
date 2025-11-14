import os
from pathlib import Path
from typing import List

def get_jpeg_files(folder_path: str) -> List[Path]:
    """Возвращает список JPEG файлов в указанной папке"""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
    image_files = []
    
    for extension in extensions:
        image_files.extend(folder.glob(extension))
    
    return sorted(image_files)

def create_output_folder(base_path: str, suffix: str = "_cropped") -> str:
    """Создает папку для сохранения результатов"""
    base = Path(base_path)
    output_folder = base.parent / f"{base.name}{suffix}"
    output_folder.mkdir(exist_ok=True)
    return str(output_folder)
