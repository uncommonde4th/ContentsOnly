import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import logging
import sys
import os

# Добавляем src в путь для импортов
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..')
sys.path.insert(0, src_path)

from src.scanner.image_processor import ImageProcessor
from src.utils.config import ProcessingConfig

class MainWindow:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Создание пользовательского интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка расширения
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame, 
            text="Автоматическая обрезка документов", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Выбор входной папки
        ttk.Label(main_frame, text="Входная папка:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_folder_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_folder_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Обзор", command=self.browse_input_folder).grid(row=1, column=2, padx=5)
        
        # Выбор выходной папки
        ttk.Label(main_frame, text="Выходная папка:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_folder_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.output_folder_var, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Обзор", command=self.browse_output_folder).grid(row=2, column=2, padx=5)
        
        # Настройки обработки
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки обработки", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        settings_frame.columnconfigure(1, weight=1)
        
        # Качество JPEG
        ttk.Label(settings_frame, text="Качество JPEG:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.quality_var = tk.IntVar(value=95)
        quality_scale = ttk.Scale(
            settings_frame, 
            from_=50, to=100, 
            variable=self.quality_var,
            orient=tk.HORIZONTAL
        )
        quality_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.quality_label = ttk.Label(settings_frame, text="95%")
        self.quality_label.grid(row=0, column=2, padx=5)
        quality_scale.configure(command=self.on_quality_change)
        
        # Выравнивание перспективы
        self.perspective_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame, 
            text="Выравнивать перспективу", 
            variable=self.perspective_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Кнопка обработки
        self.process_button = ttk.Button(
            main_frame, 
            text="Начать обработку", 
            command=self.start_processing
        )
        self.process_button.grid(row=4, column=0, columnspan=3, pady=20)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Статус
        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=6, column=0, columnspan=3, pady=5)
        
    def on_quality_change(self, value):
        """Обновление значения качества"""
        quality = int(float(value))
        self.quality_label.config(text=f"{quality}%")
        
    def browse_input_folder(self):
        """Выбор входной папки"""
        folder = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder:
            self.input_folder_var.set(folder)
            # Автоматически создаем выходную папку
            input_path = Path(folder)
            output_path = input_path.parent / f"{input_path.name}_cropped"
            self.output_folder_var.set(str(output_path))
            
    def browse_output_folder(self):
        """Выбор выходной папки"""
        folder = filedialog.askdirectory(title="Выберите папку для сохранения")
        if folder:
            self.output_folder_var.set(folder)
            
    def start_processing(self):
        """Запуск обработки в отдельном потоке"""
        if not self.input_folder_var.get():
            messagebox.showerror("Ошибка", "Выберите входную папку")
            return
            
        # Обновление конфигурации
        self.config.jpeg_quality = self.quality_var.get()
        self.config.enable_perspective_correction = self.perspective_var.get()
        
        # Запуск в отдельном потоке
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
        
    def process_images(self):
        """Обработка изображений (в отдельном потоке)"""
        try:
            self.set_processing_state(False)
            self.progress.start()
            self.status_var.set("Обработка...")
            
            processor = ImageProcessor(self.config)
            stats = processor.process_folder(
                self.input_folder_var.get(),
                self.output_folder_var.get()
            )
            
            self.progress.stop()
            self.set_processing_state(True)
            
            # Показ результатов
            messagebox.showinfo(
                "Обработка завершена",
                f"Обработано: {stats['processed']}/{stats['total']} файлов\n"
                f"Ошибок: {stats['failed']}\n"
                f"Результаты сохранены в: {self.output_folder_var.get()}"
            )
            self.status_var.set("Обработка завершена")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки: {str(e)}")
            self.progress.stop()
            self.set_processing_state(True)
            messagebox.showerror("Ошибка", f"Произошла ошибка при обработке: {str(e)}")
            self.status_var.set("Ошибка обработки")
    
    def set_processing_state(self, enabled):
        """Включение/выключение элементов управления во время обработки"""
        state = "normal" if enabled else "disabled"
        self.process_button.config(state=state)
