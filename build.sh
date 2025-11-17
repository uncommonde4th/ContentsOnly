#!/bin/bash

echo "Сборка ContentsOnly..."

# Создание виртуального окружения
python -m venv build_venv
source build_venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt
pip install cx_Freeze

# Сборка приложения
python setup.py build

echo ""
echo "Сборка завершена!"
echo "Исполняемый файл находится в папке: build/"
