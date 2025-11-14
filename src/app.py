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

class DocumentScannerApp:
    def __init__(self):
        self.processing_config = ProcessingConfig()
        self.calibration_config = CalibrationConfig()
        self.calibration_manager = CalibrationManager(self.calibration_config)
        self.current_calibration_image = None
        self.setup_gui()
    
    def setup_gui(self):
        """Создает GUI с калибровкой"""
        self.root = tk.Tk()
        self.root.title("Document Scanner with Calibration")
        self.root.geometry("800x700")
        
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка калибровки
        self.calibration_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.calibration_frame, text='🎯 Калибровка')
        
        # Вкладка обработки
        self.processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_frame, text='⚡ Обработка')
        
        self.setup_calibration_tab()
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
        
        self.progress_var = tk.StringVar(value="")
        ttk.Label(main_frame, textvariable=self.progress_var).grid(row=4, column=0, columnspan=3)
    
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
            
            processor = CalibratedImageProcessor(self.processing_config, self.calibration_config)
            stats = processor.process_folder(self.process_input_var.get(), self.process_output_var.get())
            
            self.process_btn.config(state='normal')
            self.progress_var.set("")
            
            messagebox.showinfo("Готово!", 
                              f"Обработано: {stats['processed']} файлов\n"
                              f"Папка с результатами:\n{self.process_output_var.get()}")
            
        except Exception as e:
            self.process_btn.config(state='normal')
            self.progress_var.set("")
            messagebox.showerror("Ошибка", f"Ошибка обработки: {str(e)}")

def main():
    app = DocumentScannerApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()
