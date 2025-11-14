import sys
import os
import logging
from pathlib import Path

# Добавляем src в Python path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.gui.main_window import MainWindow
from src.utils.config import ProcessingConfig
import tkinter as tk
from tkinter import messagebox

class DocumentScannerApp:
    """Главный класс приложения"""
    
    def __init__(self):
        self.setup_logging()
        self.config = ProcessingConfig()
        self.root = None
        
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = Path.home() / ".document_scanner"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "app.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Запуск приложения"""
        try:
            self.logger.info("Запуск Document Scanner App")
            
            # Создаем главное окно
            self.root = tk.Tk()
            self.root.title("Document Scanner - Автоматическая обрезка документов")
            self.root.geometry("800x600")
            
            # Создаем главное окно
            app_window = MainWindow(self.root, self.config)
            
            # Обработка закрытия окна
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Запуск главного цикла
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Ошибка при запуске приложения: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось запустить приложение: {str(e)}")
    
    def on_closing(self):
        """Обработчик закрытия приложения"""
        self.logger.info("Завершение работы приложения")
        self.root.destroy()

def main():
    """Точка входа приложения"""
    app = DocumentScannerApp()
    app.run()

if __name__ == "__main__":
    main()
