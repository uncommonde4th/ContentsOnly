#!/bin/bash

# build_simple_auto.sh - Автосборка для Windows и Linux

set -e

echo "🛠️  Автоматическая сборка ContentsOnly для Windows и Linux..."
echo "========================================================"

# Создаем папки релизов
mkdir -p release/windows release/linux

# Функция для сборки Windows версии
build_windows() {
    echo "🍷 Сборка Windows версии..."
    
    export WINEPREFIX="$HOME/.wine_contentsonly_simple"
    
    # Если Wine prefix не существует, создаем
    if [ ! -d "$WINEPREFIX" ]; then
        echo "Инициализация Wine..."
        wineboot -i
        sleep 10
    fi
    
    # Проверяем установлен ли Python
    if ! wine python --version 2>/dev/null; then
        echo "❌ Python для Windows не установлен!"
        echo "📥 Скачиваю установщик Python..."
        wget -O python_installer.exe "https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe"
        
        echo "🚀 Запускаю установку Python..."
        echo "================================================"
        echo "Установите Python в стандартную папку C:\Python38"
        echo "Обязательно отметьте 'Add Python to PATH'"
        echo "После установки закройте установщик и нажмите Enter здесь"
        echo "================================================"
        wine python_installer.exe
        read -p "Нажмите Enter после завершения установки Python..."
    fi
    
    echo "📦 Устанавливаю зависимости для Windows..."
    wine pip install --upgrade pip
    wine pip install opencv-python numpy Pillow cx_Freeze
    
    echo "🔨 Собираю Windows версию..."
    wine python setup_win.py build
    
    if [ -d "build" ]; then
        echo "📦 Упаковываю Windows версию..."
        mv build build_windows
        cd build_windows
        zip -r ../release/windows/ContentsOnly_windows_x64.zip ./*
        cd ..
        echo "✅ Windows сборка готова: release/windows/ContentsOnly_windows_x64.zip"
    else
        echo "❌ Ошибка: Windows сборка не удалась"
    fi
}

# Функция для сборки Linux версии
build_linux() {
    echo "🐧 Сборка Linux версии..."
    
    # Активируем виртуальное окружение если есть
    if [ -d "build_venv" ]; then
        echo "🔧 Активация виртуального окружения..."
        source build_venv/bin/activate
    else
        echo "🔧 Создание виртуального окружения..."
        python -m venv build_venv
        source build_venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install cx_Freeze
    fi
    
    echo "🔨 Собираю Linux версию..."
    python setup.py build
    
    if [ -d "build" ]; then
        echo "📦 Упаковываю Linux версию..."
        # Создаем архив с бинарником
        tar -czf release/linux/ContentsOnly_linux_x86_64.tar.gz build/
        
        # Также создаем отдельный архив только с исполняемым файлом для удобства
        cd build/exe.linux-*
        tar -czf ../../release/linux/ContentsOnly_standalone_linux_x86_64.tar.gz ContentsOnly
        cd ../..
        
        echo "✅ Linux сборка готова: release/linux/ContentsOnly_linux_x86_64.tar.gz"
    else
        echo "❌ Ошибка: Linux сборка не удалась"
    fi
}

# Функция создания README файлов
create_readme() {
    echo "📝 Создаю README файлы..."
    
    # README для Windows
    cat > release/windows/README_Windows.txt << 'EOF'
ContentsOnly - Document Scanner для Windows
===========================================

📥 Установка:
1. Распакуйте архив ContentsOnly_windows_x64.zip
2. Перейдите в папку build/exe.win-amd64-3.8/
3. Запустите ContentsOnly.exe

🖼️ Использование:
- Загрузите фотографию документа
- Программа автоматически обнаружит документ
- Нажмите "Сохранить" для экспорта

❓ Поддержка:
Если программа не запускается, убедитесь что установлены:
- Visual C++ Redistributable
- .NET Framework 4.5+

EOF

    # README для Linux
    cat > release/linux/README_Linux.txt << 'EOF'
ContentsOnly - Document Scanner для Linux
=========================================

📥 Установка:
1. Распакуйте архив: tar -xzf ContentsOnly_linux_x86_64.tar.gz
2. Перейдите в папку: cd build/exe.linux-x86_64-3.*/
3. Запустите: ./ContentsOnly

🖼️ Использование:
- Загрузите фотографию документа
- Программа автоматически обнаружит документ
- Нажмите "Сохранить" для экспорта

❓ Поддержка:
Если программа не запускается, установите зависимости:
sudo apt-get install python3-tk python3-opencv
# или для Manjaro/Arch:
sudo pacman -S tk opencv

EOF
}

# Функция создания setup_win.py если нет
create_setup_win() {
    if [ ! -f "setup_win.py" ]; then
        echo "📄 Создаю setup_win.py..."
        cat > setup_win.py << 'EOF'
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
EOF
    fi
}

# Основной процесс сборки
echo "🚀 Начинаю сборку..."

# Создаем setup_win.py если нужно
create_setup_win

# Собираем Windows версию
build_windows

# Собираем Linux версию  
build_linux

# Создаем README файлы
create_readme

# Создаем общий README
cat > release/README.md << 'EOF'
# ContentsOnly - Document Scanner

Автоматическая обрезка документов из фотографий.

## 📦 Версии

### Windows
- **Файл**: `windows/ContentsOnly_windows_x64.zip`
- **Запуск**: Распакуйте и запустите `ContentsOnly.exe`

### Linux  
- **Файл**: `linux/ContentsOnly_linux_x86_64.tar.gz`
- **Запуск**: `tar -xzf` затем `./ContentsOnly`

## 🖼️ Использование
1. Загрузите фотографию документа
2. Программа автоматически обнаружит границы
3. Сохраните результат

## 📋 Системные требования
- **Windows**: 7/10/11, 2GB RAM
- **Linux**: Ubuntu 18.04+, Manjaro, 2GB RAM

EOF

echo ""
echo "🎉 Сборка завершена!"
echo "========================================================"
echo "📁 Созданные файлы:"
echo ""
echo "Windows:"
ls -lh release/windows/
echo ""
echo "Linux:"
ls -lh release/linux/
echo ""
echo "📦 Итоговые архивы для распространения:"
echo "  - release/windows/ContentsOnly_windows_x64.zip"
echo "  - release/linux/ContentsOnly_linux_x86_64.tar.gz"
echo ""
echo "🚀 Для GitHub релиза скопируйте оба файла!"
