import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np

# Импортируем наши модули
from scanner.calibration import CalibrationManager, CalibrationConfig
from scanner.image_processor import CalibratedImageProcessor, ProcessingConfig
from scanner.manual_crop import ManualCropManager, ManualCropConfig

class DocumentScannerApp:
    def __init__(self):
        self.processing_config = ProcessingConfig()
        self.calibration_config = CalibrationConfig()
        self.calibration_manager = CalibrationManager(self.calibration_config)
        self.current_calibration_image = None
        
        # Ручная обрезка
        self.manual_crop_config = ManualCropConfig()
        self.manual_crop_manager = ManualCropManager(self.manual_crop_config, self.calibration_config)
        self.current_manual_crop_image = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """Создает GUI с калибровкой"""
        self.root = tk.Tk()
        self.root.title("Document Scanner with Calibration")
        self.root.geometry("1600x1400")
        
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка калибровки
        self.calibration_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calibration_frame, text='🎯 Калибровка')
        
        # Вкладка ручной обрезки
        self.manual_crop_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.manual_crop_frame, text='✂️ Ручная обрезка')
        
        # Вкладка обработки
        self.processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_frame, text='⚡ Обработка')
        
        self.setup_calibration_tab()
        self.setup_manual_crop_tab()
        self.setup_processing_tab()
        
        # Обновляем статус калибровки
        self.update_calibration_status()
    
    def setup_calibration_tab(self):
        """Настраивает вкладку калибровки"""
        # Верхняя панель
        top_frame = ttk.Frame(self.calibration_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(top_frame, text="Папка для калибровки:").grid(row=0, column=0, sticky='w')
        self.calib_input_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.calib_input_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="📁", command=self.browse_calibration_folder, width=3).grid(row=0, column=2)
        
        ttk.Button(top_frame, text="🔄 Загрузить изображения", 
                  command=self.load_calibration_images).grid(row=1, column=0, columnspan=3, pady=10)
        
        # Область изображения
        self.image_frame = ttk.Frame(self.calibration_frame)
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.image_frame, bg='gray')
        self.canvas.pack(fill='both', expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Нижняя панель управления
        bottom_frame = ttk.Frame(self.calibration_frame)
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.calib_status_var = tk.StringVar(value="Выберите папку и загрузите изображения")
        ttk.Label(bottom_frame, textvariable=self.calib_status_var).grid(row=0, column=0, columnspan=2, sticky='w')
        
        ttk.Button(bottom_frame, text="↶ Удалить последнюю точку", 
                  command=self.remove_last_point).grid(row=1, column=0, pady=5)
        ttk.Button(bottom_frame, text="🗑️ Очистить все точки", 
                  command=self.clear_points).grid(row=1, column=1, pady=5)
        ttk.Button(bottom_frame, text="💾 Сохранить калибровку", 
                  command=self.save_calibration).grid(row=2, column=0, pady=5)
        ttk.Button(bottom_frame, text="⏭️ Следующее изображение", 
                  command=self.next_calibration_image).grid(row=2, column=1, pady=5)
    
    def setup_manual_crop_tab(self):
        """Настраивает вкладку ручной обрезки"""
        # Верхняя панель
        top_frame = ttk.Frame(self.manual_crop_frame)
        top_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(top_frame, text="Папка с изображениями:").grid(row=0, column=0, sticky='w')
        self.manual_crop_input_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.manual_crop_input_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top_frame, text="📁", command=self.browse_manual_crop_folder, width=3).grid(row=0, column=2)
        
        ttk.Label(top_frame, text="Папка для обрезанных:").grid(row=1, column=0, sticky='w', pady=5)
        self.manual_crop_output_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.manual_crop_output_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(top_frame, text="📁", command=self.browse_manual_crop_output, width=3).grid(row=1, column=2)
        
        # Опция подсказки обрезки
        self.hint_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top_frame, text="💡 Подсказка обрезки (автоматическая установка точек)", 
                       variable=self.hint_enabled_var,
                       command=self.on_hint_toggle).grid(row=2, column=0, columnspan=3, sticky='w', pady=5)
        
        ttk.Button(top_frame, text="🔄 Загрузить изображения", 
                  command=self.load_manual_crop_images).grid(row=3, column=0, columnspan=3, pady=10)
        
        # Область изображения
        self.manual_crop_image_frame = ttk.Frame(self.manual_crop_frame)
        self.manual_crop_image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.manual_crop_canvas = tk.Canvas(self.manual_crop_image_frame, bg='gray')
        self.manual_crop_canvas.pack(fill='both', expand=True)
        
        # Привязываем события для интерактивных точек
        self.manual_crop_canvas.bind("<Button-1>", self.on_manual_crop_canvas_click)
        self.manual_crop_canvas.bind("<B1-Motion>", self.on_manual_crop_canvas_drag)
        self.manual_crop_canvas.bind("<ButtonRelease-1>", self.on_manual_crop_canvas_release)
        self.manual_crop_canvas.bind("<Motion>", self.on_manual_crop_canvas_motion)
        
        # Нижняя панель управления
        bottom_frame = ttk.Frame(self.manual_crop_frame)
        bottom_frame.pack(fill='x', padx=10, pady=10)
        
        self.manual_crop_status_var = tk.StringVar(value="Выберите папку и загрузите изображения")
        ttk.Label(bottom_frame, textvariable=self.manual_crop_status_var).grid(row=0, column=0, columnspan=3, sticky='w')
        
        # Кнопки управления
        button_frame = ttk.Frame(bottom_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="↶ Удалить последнюю точку", 
                  command=self.remove_last_manual_point).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="🗑️ Очистить все точки", 
                  command=self.clear_manual_points).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="💡 Применить подсказку", 
                  command=self.apply_hint).grid(row=0, column=2, padx=5)
        
        ttk.Button(button_frame, text="⏮️ Предыдущее", 
                  command=self.previous_manual_crop_image).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(button_frame, text="💾 Сохранить результат", 
                  command=self.save_manual_crop).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(button_frame, text="⏭️ Следующее", 
                  command=self.next_manual_crop_image).grid(row=1, column=2, padx=5, pady=5)
    
    def setup_processing_tab(self):
        """Настраивает вкладку обработки"""
        main_frame = ttk.Frame(self.processing_frame, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Папка для обработки:", font=("Arial", 12)).grid(row=0, column=0, sticky='w', pady=5)
        self.process_input_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.process_input_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="📁", command=self.browse_process_folder, width=3).grid(row=0, column=2, pady=5)
        
        ttk.Label(main_frame, text="Папка для результатов:").grid(row=1, column=0, sticky='w', pady=5)
        self.process_output_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.process_output_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="📁", command=self.browse_process_output, width=3).grid(row=1, column=2, pady=5)
        
        # Статус калибровки
        self.process_status_var = tk.StringVar(value="❌ Калибровка не выполнена")
        ttk.Label(main_frame, textvariable=self.process_status_var, font=("Arial", 10)).grid(row=2, column=0, columnspan=3, pady=10)
        
        self.process_btn = ttk.Button(main_frame, text="🚀 НАЧАТЬ ОБРАБОТКУ", 
                                    command=self.start_processing, state='disabled')
        self.process_btn.grid(row=3, column=0, columnspan=3, pady=20)
        
        # Прогресс-бар
        self.progress_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=3, pady=5)
        
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate')
        self.progress_bar.grid(row=5, column=0, columnspan=3, sticky='ew', padx=20, pady=5)
        
        self.progress_filename_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.progress_filename_var, font=("Arial", 9)).grid(row=6, column=0, columnspan=3, pady=2)
    
    def browse_calibration_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку для калибровки")
        if folder:
            self.calib_input_var.set(folder)
    
    def browse_process_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку для обработки")
        if folder:
            self.process_input_var.set(folder)
            output_path = Path(folder).parent / f"{Path(folder).name}_cropped"
            self.process_output_var.set(str(output_path))
    
    def browse_process_output(self):
        folder = filedialog.askdirectory(title="Выберите папку для результатов")
        if folder:
            self.process_output_var.set(folder)
    
    def browse_manual_crop_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с изображениями для обрезки")
        if folder:
            self.manual_crop_input_var.set(folder)
            output_path = Path(folder).parent / f"{Path(folder).name}_cropped"
            self.manual_crop_output_var.set(str(output_path))
    
    def browse_manual_crop_output(self):
        folder = filedialog.askdirectory(title="Выберите папку для обрезанных изображений")
        if folder:
            self.manual_crop_output_var.set(folder)
    
    def load_calibration_images(self):
        """Загружает изображения для калибровки"""
        if not self.calib_input_var.get():
            messagebox.showerror("Ошибка", "Выберите папку для калибровки!")
            return
        
        if self.calibration_manager.load_images_from_folder(self.calib_input_var.get()):
            self.next_calibration_image()
        else:
            messagebox.showerror("Ошибка", "В папке нет изображений для калибровки!")
    
    def next_calibration_image(self):
        """Загружает следующее изображение для калибровки"""
        result = self.calibration_manager.get_next_calibration_image()
        if result is None:
            messagebox.showinfo("Информация", "Все изображения для калибровки просмотрены!")
            return
        
        image, filename = result
        self.current_calibration_image = image
        self.display_calibration_image(image)
        
        current, total = self.calibration_manager.get_progress()
        self.calib_status_var.set(f"Изображение {current}/{total}: {filename} - Отметьте 4 угла документа")
    
    def display_calibration_image(self, image: np.ndarray):
        """Отображает изображение на canvas"""
        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получаем аннотированное изображение
        annotated_image = self.calibration_manager.get_annotated_image()
        if annotated_image is not None:
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Масштабируем для отображения
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Масштабируем сохраняя пропорции
            img_ratio = pil_image.width / pil_image.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor='center')
        
        # Сохраняем размеры для преобразования координат
        self.display_scale_x = image.shape[1] / pil_image.width
        self.display_scale_y = image.shape[0] / pil_image.height
        self.display_offset_x = (canvas_width - pil_image.width) // 2
        self.display_offset_y = (canvas_height - pil_image.height) // 2
    
    def on_canvas_click(self, event):
        """Обработчик клика по canvas"""
        if self.current_calibration_image is None:
            return
        
        # Убеждаемся что размеры для преобразования координат инициализированы
        if not hasattr(self, 'display_scale_x') or not hasattr(self, 'display_scale_y'):
            # Если еще не инициализированы, обновляем отображение
            self.display_calibration_image(self.current_calibration_image)
        
        # Преобразуем координаты canvas в координаты изображения
        x_img = int((event.x - self.display_offset_x) * self.display_scale_x)
        y_img = int((event.y - self.display_offset_y) * self.display_scale_y)
        
        # Проверяем что клик внутри изображения
        if (0 <= x_img < self.current_calibration_image.shape[1] and 
            0 <= y_img < self.current_calibration_image.shape[0]):
            
            points_before = len(self.calibration_manager.current_points)
            self.calibration_manager.add_point(x_img, y_img)
            points_after = len(self.calibration_manager.current_points)
            
            # Обновляем отображение
            self.display_calibration_image(self.current_calibration_image)
            
            # Показываем сообщение только если добавили 4-ю точку (было 3, стало 4)
            if points_before == 3 and points_after == 4:
                messagebox.showinfo("Успех", "4 точки отмечены! Сохраните калибровку или перейдите к следующему изображению.")
    
    def remove_last_point(self):
        """Удаляет последнюю точку"""
        self.calibration_manager.remove_last_point()
        if self.current_calibration_image:
            self.display_calibration_image(self.current_calibration_image)
    
    def clear_points(self):
        """Очищает все точки"""
        self.calibration_manager.clear_points()
        if self.current_calibration_image:
            self.display_calibration_image(self.current_calibration_image)
    
    def save_calibration(self):
        """Сохраняет калибровку"""
        if self.calibration_manager.save_calibration():
            messagebox.showinfo("Успех", "Калибровка сохранена! Теперь можно обрабатывать изображения.")
            self.update_calibration_status()
        else:
            messagebox.showerror("Ошибка", "Нужно отметить 4 точки для сохранения калибровки!")
    
    def update_calibration_status(self):
        """Обновляет статус калибровки"""
        if self.calibration_manager.is_complete():
            self.process_status_var.set("✅ Калибровка выполнена")
            self.process_btn.config(state='normal')
        else:
            self.process_status_var.set("❌ Калибровка не выполнена")
            self.process_btn.config(state='disabled')
    
    def start_processing(self):
        """Запускает обработку изображений"""
        if not self.calibration_manager.is_complete():
            messagebox.showerror("Ошибка", "Сначала выполните калибровку!")
            return
        
        if not self.process_input_var.get():
            messagebox.showerror("Ошибка", "Выберите папку для обработки!")
            return
        
        # Переключаем на вкладку обработки
        self.notebook.select(1)
        
        # Запускаем в отдельном потоке
        thread = threading.Thread(target=self.process_images)
        thread.daemon = True
        thread.start()
    
    def process_images(self):
        """Обрабатывает изображения"""
        try:
            self.process_btn.config(state='disabled')
            self.progress_var.set("Обработка...")
            self.progress_bar['maximum'] = 100
            self.progress_bar['value'] = 0
            self.progress_filename_var.set("")
            
            def update_progress(current, total, filename):
                """Обновляет прогресс-бар"""
                progress = int((current / total) * 100) if total > 0 else 0
                self.progress_bar['value'] = progress
                self.progress_var.set(f"Обработка: {current}/{total} ({progress}%)")
                self.progress_filename_var.set(f"Текущий файл: {filename}")
                self.root.update_idletasks()  # Обновляем GUI
            
            processor = CalibratedImageProcessor(self.processing_config, self.calibration_config)
            stats = processor.process_folder(
                self.process_input_var.get(), 
                self.process_output_var.get(),
                calibration_manager=self.calibration_manager,
                progress_callback=update_progress
            )
            
            self.process_btn.config(state='normal')
            self.progress_var.set("")
            self.progress_bar['value'] = 100
            self.progress_filename_var.set("")
            
            messagebox.showinfo("Готово!", 
                              f"Обработано: {stats['processed']} файлов\n"
                              f"Ошибок: {stats['failed']}\n"
                              f"Папка с результатами:\n{self.process_output_var.get()}")
            
        except Exception as e:
            self.process_btn.config(state='normal')
            self.progress_var.set("")
            self.progress_bar['value'] = 0
            self.progress_filename_var.set("")
            messagebox.showerror("Ошибка", f"Ошибка обработки: {str(e)}")
    
    # Методы для ручной обрезки
    def load_manual_crop_images(self):
        """Загружает изображения для ручной обрезки"""
        if not self.manual_crop_input_var.get():
            messagebox.showerror("Ошибка", "Выберите папку с изображениями!")
            return
        
        if self.manual_crop_manager.load_images_from_folder(self.manual_crop_input_var.get()):
            self.next_manual_crop_image()
        else:
            messagebox.showerror("Ошибка", "В папке нет изображений для обрезки!")
    
    def next_manual_crop_image(self):
        """Загружает следующее изображение для ручной обрезки (пропускает текущее без сохранения)"""
        # Пропускаем текущее изображение без сохранения
        self.manual_crop_manager.skip_current_image()
        
        result = self.manual_crop_manager.get_next_image()
        if result is None:
            messagebox.showinfo("Информация", "Все изображения обработаны!")
            return
        
        image, filename = result
        self.current_manual_crop_image = image
        
        # Сначала отображаем изображение
        self.display_manual_crop_image(image)
        
        # Затем применяем подсказку если включена (после небольшой задержки для инициализации canvas)
        if self.hint_enabled_var.get():
            self.root.after(100, self.apply_hint)
        
        current, total = self.manual_crop_manager.get_progress()
        self.manual_crop_status_var.set(f"Обработано: {current}/{total} | Текущее: {filename}")
    
    def previous_manual_crop_image(self):
        """Загружает предыдущее изображение"""
        result = self.manual_crop_manager.get_previous_image()
        if result is None:
            messagebox.showinfo("Информация", "Это первое изображение!")
            return
        
        image, filename = result
        self.current_manual_crop_image = image
        
        # Сначала отображаем изображение
        self.display_manual_crop_image(image)
        
        # Затем применяем подсказку если включена (после небольшой задержки для инициализации canvas)
        if self.hint_enabled_var.get():
            self.root.after(100, self.apply_hint)
        
        current, total = self.manual_crop_manager.get_progress()
        self.manual_crop_status_var.set(f"Обработано: {current}/{total} | Текущее: {filename}")
    
    def display_manual_crop_image(self, image: np.ndarray):
        """Отображает изображение на canvas для ручной обрезки"""
        # Обновляем canvas чтобы получить правильные размеры
        self.manual_crop_canvas.update_idletasks()
        
        # Конвертируем BGR в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получаем аннотированное изображение
        annotated_image = self.manual_crop_manager.get_annotated_image()
        if annotated_image is not None:
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Масштабируем для отображения
        canvas_width = self.manual_crop_canvas.winfo_width()
        canvas_height = self.manual_crop_canvas.winfo_height()
        
        # Если canvas еще не инициализирован, используем размеры по умолчанию
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400
        
        # Масштабируем сохраняя пропорции
        img_ratio = pil_image.width / pil_image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.manual_crop_photo = ImageTk.PhotoImage(pil_image)
        self.manual_crop_canvas.delete("all")
        self.manual_crop_canvas.create_image(canvas_width // 2, canvas_height // 2, 
                                            image=self.manual_crop_photo, anchor='center')
        
        # Сохраняем размеры для преобразования координат
        self.manual_crop_display_scale_x = image.shape[1] / pil_image.width
        self.manual_crop_display_scale_y = image.shape[0] / pil_image.height
        self.manual_crop_display_offset_x = (canvas_width - pil_image.width) // 2
        self.manual_crop_display_offset_y = (canvas_height - pil_image.height) // 2
    
    def on_manual_crop_canvas_click(self, event):
        """Обработчик клика по canvas для ручной обрезки"""
        if self.current_manual_crop_image is None:
            return
        
        # Убеждаемся что размеры для преобразования координат инициализированы
        if not hasattr(self, 'manual_crop_display_scale_x') or not hasattr(self, 'manual_crop_display_scale_y'):
            self.display_manual_crop_image(self.current_manual_crop_image)
        
        # Преобразуем координаты canvas в координаты изображения
        x_img = int((event.x - self.manual_crop_display_offset_x) * self.manual_crop_display_scale_x)
        y_img = int((event.y - self.manual_crop_display_offset_y) * self.manual_crop_display_scale_y)
        
        # Проверяем что клик внутри изображения
        if (0 <= x_img < self.current_manual_crop_image.shape[1] and 
            0 <= y_img < self.current_manual_crop_image.shape[0]):
            
            # Пробуем начать перетаскивание существующей точки
            if not self.manual_crop_manager.start_drag(x_img, y_img):
                # Если не попали в точку, добавляем новую (если их меньше 4)
                if len(self.manual_crop_manager.current_points) < 4:
                    self.manual_crop_manager.add_point(x_img, y_img)
            
            # Обновляем отображение
            self.display_manual_crop_image(self.current_manual_crop_image)
    
    def on_manual_crop_canvas_drag(self, event):
        """Обработчик перетаскивания на canvas"""
        if self.current_manual_crop_image is None:
            return
        
        # Преобразуем координаты
        x_img = int((event.x - self.manual_crop_display_offset_x) * self.manual_crop_display_scale_x)
        y_img = int((event.y - self.manual_crop_display_offset_y) * self.manual_crop_display_scale_y)
        
        # Обновляем позицию точки
        self.manual_crop_manager.update_drag(x_img, y_img)
        
        # Обновляем отображение
        self.display_manual_crop_image(self.current_manual_crop_image)
    
    def on_manual_crop_canvas_release(self, event):
        """Обработчик отпускания кнопки мыши"""
        self.manual_crop_manager.end_drag()
        if self.current_manual_crop_image is not None:
            self.display_manual_crop_image(self.current_manual_crop_image)
    
    def on_manual_crop_canvas_motion(self, event):
        """Обработчик движения мыши для подсветки точек"""
        if self.current_manual_crop_image is None:
            return
        
        # Преобразуем координаты
        x_img = int((event.x - self.manual_crop_display_offset_x) * self.manual_crop_display_scale_x)
        y_img = int((event.y - self.manual_crop_display_offset_y) * self.manual_crop_display_scale_y)
        
        # Сохраняем предыдущий hover индекс
        old_hover = self.manual_crop_manager.hover_point_index
        
        # Обновляем hover
        self.manual_crop_manager.update_hover(x_img, y_img)
        
        # Обновляем отображение только если изменился hover
        if old_hover != self.manual_crop_manager.hover_point_index:
            if self.current_manual_crop_image is not None:
                self.display_manual_crop_image(self.current_manual_crop_image)
    
    def remove_last_manual_point(self):
        """Удаляет последнюю точку"""
        self.manual_crop_manager.remove_last_point()
        if self.current_manual_crop_image is not None:
            self.display_manual_crop_image(self.current_manual_crop_image)
    
    def clear_manual_points(self):
        """Очищает все точки"""
        self.manual_crop_manager.clear_points()
        if self.current_manual_crop_image is not None:
            self.display_manual_crop_image(self.current_manual_crop_image)
    
    def apply_hint(self):
        """Применяет подсказку обрезки"""
        if self.current_manual_crop_image is None:
            return
        
        suggested_points = self.manual_crop_manager.get_suggested_points()
        if suggested_points:
            self.manual_crop_manager.set_points(suggested_points)
            self.display_manual_crop_image(self.current_manual_crop_image)
        # Не показываем сообщение если подсказка недоступна - это нормально
    
    def on_hint_toggle(self):
        """Обработчик переключения опции подсказки"""
        # Если включили подсказку и есть текущее изображение, применяем
        if self.hint_enabled_var.get() and self.current_manual_crop_image is not None:
            if len(self.manual_crop_manager.current_points) == 0:
                self.apply_hint()
            else:
                # Если точки уже есть, переприменяем подсказку
                self.apply_hint()
    
    def save_manual_crop(self):
        """Сохраняет обрезанное изображение"""
        if len(self.manual_crop_manager.current_points) != 4:
            messagebox.showerror("Ошибка", "Нужно отметить 4 точки для сохранения!")
            return
        
        if not self.manual_crop_output_var.get():
            messagebox.showerror("Ошибка", "Выберите папку для сохранения!")
            return
        
        # Получаем имя текущего файла
        if self.manual_crop_manager.current_index > 0:
            current_path = self.manual_crop_manager.image_paths[self.manual_crop_manager.current_index - 1]
            filename = Path(current_path).name
            output_path = Path(self.manual_crop_output_var.get()) / filename
        else:
            messagebox.showerror("Ошибка", "Нет текущего изображения!")
            return
        
        # Сохраняем
        if self.manual_crop_manager.save_crop(str(output_path)):
            # Переходим к следующему изображению
            result = self.manual_crop_manager.get_next_image()
            if result is None:
                messagebox.showinfo("Информация", "Все изображения обработаны!")
                return
            
            image, filename = result
            self.current_manual_crop_image = image
            
            # Сначала отображаем изображение
            self.display_manual_crop_image(image)
            
            # Затем применяем подсказку если включена
            if self.hint_enabled_var.get():
                self.root.after(100, self.apply_hint)
            
            current, total = self.manual_crop_manager.get_progress()
            self.manual_crop_status_var.set(f"Обработано: {current}/{total} | Текущее: {filename}")
        else:
            messagebox.showerror("Ошибка", "Не удалось сохранить изображение!")

def main():
    app = DocumentScannerApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()
