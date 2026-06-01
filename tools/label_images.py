#!/usr/bin/env python3
"""
Инструмент для разметки углов документов для обучения нейросети
Логика:
1. Аннотации сохраняются в labeled_data/annotations/labelme/
2. Обработанные изображения копируются в labeled_data/images/
3. При сохранении проверяется наличие обоих файлов (JSON и изображение)
4. Кнопка "Следующее" сохраняет аннотацию и загружает следующее изображение
"""

import sys
from pathlib import Path

# Добавляем корневую папку проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from scanner.manual_crop import ManualCropManager, ManualCropConfig
from scanner.calibration import CalibrationConfig

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import json
import shutil
from PIL import Image, ImageTk


class DataLabelingTool:
    """Инструмент для разметки данных для обучения нейросети"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Разметка данных для нейросети")
        self.root.geometry("1200x800")
        
        # Настройка путей по умолчанию
        self.base_data_path = Path(project_root / "labeled_data")
        self.annotations_path = self.base_data_path / "annotations" / "labelme"
        self.images_path = self.base_data_path / "images"
        
        # Создаём папки если их нет
        self.annotations_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(parents=True, exist_ok=True)
        
        self.manual_crop_config = ManualCropConfig()
        self.calibration_config = CalibrationConfig()
        self.manual_crop_manager = ManualCropManager(
            self.manual_crop_config, 
            self.calibration_config
        )
        
        self.current_image = None
        self.current_filename = None
        self.photo = None
        self.display_scale_x = 1
        self.display_scale_y = 1
        self.display_offset_x = 0
        self.display_offset_y = 0
        
        self.setup_gui()
    
    def setup_gui(self):
        # Верхняя панель
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill='x')
        
        # Информация о путях
        info_frame = ttk.LabelFrame(top_frame, text="Пути сохранения", padding="5")
        info_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        ttk.Label(info_frame, text=f"📁 Изображения: {self.images_path}", 
                 font=("Arial", 9)).grid(row=0, column=0, sticky='w')
        ttk.Label(info_frame, text=f"📄 Аннотации: {self.annotations_path}", 
                 font=("Arial", 9)).grid(row=1, column=0, sticky='w')
        
        # Выбор папки с исходными изображениями
        ttk.Label(top_frame, text="📂 Папка с исходными изображениями:").grid(row=1, column=0, sticky='w', pady=5)
        self.input_var = tk.StringVar()
        input_entry = ttk.Entry(top_frame, textvariable=self.input_var, width=60)
        input_entry.grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(top_frame, text="📁 Выбрать", command=self.browse_folder).grid(row=1, column=2, padx=5)
        
        ttk.Button(top_frame, text="🔄 Загрузить изображения", 
                  command=self.load_images).grid(row=2, column=0, columnspan=3, pady=10)
        
        # Статистика
        stats_frame = ttk.Frame(top_frame)
        stats_frame.grid(row=3, column=0, columnspan=3, pady=5, sticky='ew')
        
        self.total_annotated_var = tk.StringVar(value="📊 Размечено: 0 изображений")
        ttk.Label(stats_frame, textvariable=self.total_annotated_var).pack(side='left', padx=10)
        
        # Область изображения
        self.canvas = tk.Canvas(self.root, bg='gray', cursor='cross')
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Статус
        self.status_var = tk.StringVar(value="✅ Выберите папку и загрузите изображения")
        status_label = ttk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='center')
        status_label.pack(fill='x', padx=10, pady=5)
        
        # Кнопки управления
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="↶ Удалить последнюю точку", 
                  command=self.remove_point).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="🗑️ Очистить все точки", 
                  command=self.clear_points).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="💾 Сохранить и следующее", 
                  command=self.save_and_next).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="⏭️ Пропустить (без сохранения)", 
                  command=self.skip_image).pack(side='left', padx=5)
        
        # Инструкция
        instruction = """📖 Инструкция:
1. Отметьте 4 угла документа ПО ЧАСОВОЙ СТРЕЛКЕ (верхний-левый → верхний-правый → нижний-правый → нижний-левый)
2. После отметки 4 точек кнопка "Сохранить и следующее" станет активной
3. При сохранении изображение копируется в labeled_data/images/, аннотация в labeled_data/annotations/labelme/
4. Если оба файла уже существуют — пропускаем, если только один — перезаписываем
5. После сохранения автоматически загружается следующее изображение"""
        
        instruction_label = ttk.Label(self.root, text=instruction, font=("Arial", 8), justify='left')
        instruction_label.pack(pady=5, padx=10)
    
    def update_stats(self):
        """Обновляет статистику размеченных изображений"""
        if self.annotations_path.exists():
            annotated_count = len(list(self.annotations_path.glob('*.json')))
            self.total_annotated_var.set(f"📊 Размечено: {annotated_count} изображений")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку с изображениями для разметки")
        if folder:
            self.input_var.set(folder)
    
    def load_images(self):
        if not self.input_var.get():
            messagebox.showerror("Ошибка", "Выберите папку с изображениями для разметки!")
            return
        
        # Загружаем изображения в менеджер
        if self.manual_crop_manager.load_images_from_folder(self.input_var.get()):
            # Фильтруем уже обработанные изображения
            self.filter_unprocessed_images()
            self.next_image()
            self.update_stats()
        else:
            messagebox.showerror("Ошибка", "В папке нет изображений!")
    
    def filter_unprocessed_images(self):
        """Удаляет из списка уже обработанные изображения (есть и JSON, и изображение)"""
        processed_images = set()
        
        # Получаем список уже размеченных изображений
        for json_file in self.annotations_path.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                img_name = data.get('imagePath', '')
                if img_name:
                    processed_images.add(img_name)
        
        # Также проверяем по наличию файла изображения
        for img_file in self.images_path.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                processed_images.add(img_file.name)
        
        # Фильтруем список изображений в менеджере
        if processed_images:
            original_count = len(self.manual_crop_manager.image_paths)
            self.manual_crop_manager.image_paths = [
                p for p in self.manual_crop_manager.image_paths 
                if Path(p).name not in processed_images
            ]
            filtered_count = len(self.manual_crop_manager.image_paths)
            print(f"📊 Пропущено уже размеченных: {original_count - filtered_count}")
            
            # Сбрасываем индексы
            self.manual_crop_manager.current_index = 0
            self.manual_crop_manager.saved_indices.clear()
            self.manual_crop_manager.skipped_indices.clear()
    
    def next_image(self):
        """Загружает следующее изображение (без сохранения текущего)"""
        result = self.manual_crop_manager.get_next_image()
        if result is None:
            messagebox.showinfo("Готово!", "Все изображения размечены!")
            self.update_stats()
            return
        
        image, filename = result
        self.current_image = image
        self.current_filename = filename
        self.display_image(image)
        
        current, total = self.manual_crop_manager.get_progress()
        self.status_var.set(f"📷 Изображение {current}/{total}: {filename} — отметьте 4 угла документа")
    
    def skip_image(self):
        """Пропускает текущее изображение без сохранения"""
        self.manual_crop_manager.skip_current_image()
        self.next_image()
    
    def save_and_next(self):
        """Сохраняет аннотацию и переходит к следующему изображению"""
        if len(self.manual_crop_manager.current_points) != 4:
            messagebox.showerror("Ошибка", "Нужно отметить 4 угла документа перед сохранением!")
            return
        
        # Сохраняем аннотацию
        if self.save_annotation():
            # Переходим к следующему изображению
            self.next_image()
            self.update_stats()
    
    def save_annotation(self) -> bool:
        """Сохраняет аннотацию и копирует изображение"""
        try:
            h, w = self.current_image.shape[:2]
            points = self.manual_crop_manager.current_points
            
            # Упорядочиваем точки по часовой стрелке
            ordered_points = self.order_points_clockwise(points)
            
            # Пути для сохранения
            base_name = Path(self.current_filename).stem
            json_path = self.annotations_path / f"{base_name}.json"
            image_path = self.images_path / self.current_filename
            
            # Проверяем существование файлов
            json_exists = json_path.exists()
            image_exists = image_path.exists()
            
            if json_exists and image_exists:
                print(f"⏭️ Пропущено: {self.current_filename} (уже размечено)")
                messagebox.showwarning(
                    "Файл уже существует", 
                    f"Изображение {self.current_filename} уже размечено.\nПропускаем."
                )
                return False
            
            # Сохраняем JSON аннотацию
            annotation = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [{
                    "label": "document",
                    "points": [[float(x), float(y)] for x, y in ordered_points],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }],
                "imagePath": self.current_filename,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Сохранена аннотация: {json_path}")
            
            # Копируем изображение (если его ещё нет)
            if not image_exists:
                # Ищем исходное изображение
                source_image = None
                source_folder = Path(self.input_var.get())
                
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    test_path = source_folder / f"{base_name}{ext}"
                    if test_path.exists():
                        source_image = test_path
                        break
                
                if source_image:
                    shutil.copy2(source_image, image_path)
                    print(f"✅ Скопировано изображение: {image_path}")
                else:
                    # Если не нашли, пробуем из текущего изображения в памяти
                    cv2.imwrite(str(image_path), self.current_image)
                    print(f"✅ Сохранено изображение из памяти: {image_path}")
            else:
                print(f"ℹ️ Изображение уже существует: {image_path}")
            
            # Помечаем как сохранённое в менеджере
            if self.manual_crop_manager.current_index > 0:
                self.manual_crop_manager.saved_indices.add(self.manual_crop_manager.current_index - 1)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить аннотацию: {str(e)}")
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def order_points_clockwise(self, points):
        """Упорядочивает точки по часовой стрелке, начиная с верхнего-левого угла"""
        import numpy as np
        
        pts = np.array(points, dtype=np.float32)
        
        # Находим центр
        center = np.mean(pts, axis=0)
        
        # Вычисляем углы от центра
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # Сортируем по углу
        sorted_indices = np.argsort(angles)
        sorted_pts = pts[sorted_indices]
        
        # Находим верхний-левый угол (минимальная сумма x+y)
        sums = sorted_pts.sum(axis=1)
        top_left_idx = np.argmin(sums)
        
        # Переупорядочиваем начиная с верхнего-левого
        result = np.roll(sorted_pts, -top_left_idx, axis=0)
        
        return [(int(x), int(y)) for x, y in result]
    
    def display_image(self, image):
        """Отображает изображение на canvas"""
        self.canvas.update_idletasks()
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Рисуем точки если есть
        annotated = self.manual_crop_manager.get_annotated_image()
        if annotated is not None:
            image_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 800
            canvas_height = 600
        
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
        
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        self.canvas.create_image(x_center, y_center, image=self.photo, anchor='center')
        
        self.display_scale_x = image.shape[1] / pil_image.width
        self.display_scale_y = image.shape[0] / pil_image.height
        self.display_offset_x = x_center - (pil_image.width // 2)
        self.display_offset_y = y_center - (pil_image.height // 2)
    
    def on_click(self, event):
        if self.current_image is None:
            return
        
        x_img = int((event.x - self.display_offset_x) * self.display_scale_x)
        y_img = int((event.y - self.display_offset_y) * self.display_scale_y)
        
        if 0 <= x_img < self.current_image.shape[1] and 0 <= y_img < self.current_image.shape[0]:
            self.manual_crop_manager.add_point(x_img, y_img)
            self.display_image(self.current_image)
    
    def remove_point(self):
        self.manual_crop_manager.remove_last_point()
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    def clear_points(self):
        self.manual_crop_manager.clear_points()
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = DataLabelingTool()
    app.run()