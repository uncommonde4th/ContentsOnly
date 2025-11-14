@echo off
echo Сборка Document Scanner...

:: Создание виртуального окружения
python -m venv build_venv
call build_venv\Scripts\activate.bat

:: Установка зависимостей
pip install -r requirements.txt
pip install cx_Freeze

:: Сборка приложения
python setup.py build

echo.
echo Сборка завершена!
echo Исполняемый файл находится в папке: build\
echo.
pause
