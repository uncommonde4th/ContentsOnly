#!/usr/bin/env python3
import os
import re

def create_init_files():
    """Создает __init__.py файлы в нужных папках"""
    folders = [
        'src',
        'src/scanner', 
        'src/gui',
        'src/utils'
    ]
    
    for folder in folders:
        init_file = os.path.join(folder, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
            print(f'Created: {init_file}')
        else:
            print(f'Exists: {init_file}')

def update_imports_in_file(filepath):
    """Обновляет импорты в файле"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Заменяем относительные импорты на абсолютные
    updated_content = re.sub(
        r'from\s+\.\.([a-zA-Z_]+)',
        r'from src.\1',
        content
    )
    
    updated_content = re.sub(
        r'from\s+\.([a-zA-Z_]+)',
        r'from src.scanner.\1',
        updated_content
    )
    
    if content != updated_content:
        with open(filepath, 'w') as f:
            f.write(updated_content)
        print(f'Updated imports in: {filepath}')
    else:
        print(f'No changes needed: {filepath}')

def main():
    print("Fixing project structure...")
    
    # 1. Создаем __init__.py файлы
    create_init_files()
    
    # 2. Обновляем импорты в основных файлах
    files_to_update = [
        'src/scanner/document_detector.py',
        'src/scanner/image_processor.py', 
        'src/scanner/perspective_transform.py',
        'src/gui/main_window.py',
        'src/utils/config.py',
        'src/utils/file_utils.py'
    ]
    
    for filepath in files_to_update:
        if os.path.exists(filepath):
            update_imports_in_file(filepath)
        else:
            print(f'File not found: {filepath}')
    
    print("\n✅ Project structure fixed!")
    print("Now you can run: python src/app.py")

if __name__ == "__main__":
    main()
