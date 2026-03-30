#!/bin/bash

echo "Сборка ContentsOnly..."

# Проверка наличия tkinter
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "Устанавливаю tkinter..."
    sudo pacman -S tk python-tkinter
fi

# Создание папки resources если её нет
if [ ! -d "resources" ]; then
    echo "Создаю папку resources..."
    mkdir -p resources
    echo "Папка resources создана (пустая)"
fi

# Создание виртуального окружения
if [ -d "build_venv" ]; then
    rm -rf build_venv
fi

python -m venv build_venv
source build_venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || echo "Нет requirements.txt"
pip install cx_Freeze

# Сборка приложения
python setup.py build

# Поиск исполняемого файла
EXE_PATH=$(find build -name "ContentsOnly" -type f 2>/dev/null | head -1)

echo ""
echo "Сборка завершена!"
if [ -n "$EXE_PATH" ]; then
    echo "Исполняемый файл: $EXE_PATH"
    echo "Запуск: $EXE_PATH"
else
    echo "Исполняемый файл не найден. Проверьте папку build:"
    ls -la build/
fi