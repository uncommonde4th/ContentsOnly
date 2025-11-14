import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

class CalibrationConfig:
    """Конфигурация калибровки с параметрами для автоматического обнаружения"""
    def __init__(self):
        self.crop_points = None
        self.target_size = None
        
        # Параметры для автоматического обнаружения
        self.avg_color = None  # Средний цвет документа
        self.avg_bg_color = None  # Средний цвет фона
        self.color_threshold = 0  # Порог по цвету
        self.edge_threshold = 0  # Порог для детектора краев
        self.area_range = (0, 0)  # Диапазон площадей
        self.aspect_ratio_range = (0, 0)  # Диапазон соотношений сторон
        
        # Расширенная информация для лучшего распознавания
        self.document_area_pixels = 0  # Площадь документа в пикселях
        self.document_area_ratio = 0.0  # Площадь документа как доля изображения
        self.document_width = 0  # Ширина документа в пикселях
        self.document_height = 0  # Высота документа в пикселях
        self.document_aspect_ratio = 0.0  # Соотношение сторон документа
        
        # Цветовая информация документа (более детальная)
        self.document_color_mean = None  # Средний цвет
        self.document_color_std = None  # Стандартное отклонение цвета
        self.document_color_min = None  # Минимальный цвет
        self.document_color_max = None  # Максимальный цвет
        
        # Цветовая информация фона (более детальная)
        self.bg_color_mean = None  # Средний цвет фона
        self.bg_color_std = None  # Стандартное отклонение цвета фона
        self.bg_samples = []  # Образцы цвета фона из разных областей
        
        # Геометрическая информация
        self.calibration_image_size = None  # Размер калибровочного изображения
        
        self.calibrated = False
        self.calibration_samples = 0
    
    def analyze_calibration_image(self, image: np.ndarray, points: List[Tuple[int, int]]):
        """Анализирует калибровочное изображение и извлекает параметры"""
        h, w = image.shape[:2]
        
        # Сохраняем размер калибровочного изображения
        self.calibration_image_size = (w, h)
        
        # Сохраняем точки в процентах
        self.crop_points = [(x / w, y / h) for x, y in points]
        
        # Анализируем геометрию (сначала, чтобы получить размеры)
        self._analyze_geometry(points, (w, h))
        
        # Анализируем цвета (с расширенной информацией)
        self._analyze_colors(image, points)
        
        # Анализируем края и контуры
        self._analyze_edges(image, points)
        
        self.calibration_samples += 1
        self.calibrated = True
        
        print(f"🔧 Анализ калибровки завершен:")
        print(f"   - Размер изображения: {w}x{h}")
        print(f"   - Площадь документа: {self.document_area_pixels} пикселей ({self.document_area_ratio*100:.1f}%)")
        print(f"   - Размер документа: {self.document_width}x{self.document_height}")
        print(f"   - Соотношение сторон: {self.document_aspect_ratio:.2f}")
        print(f"   - Цвет документа (средний): {self.avg_color}")
        print(f"   - Цвет фона (средний): {self.avg_bg_color}")
        print(f"   - Порог цвета: {self.color_threshold}")
        print(f"   - Диапазон площади: {self.area_range}")
        print(f"   - Диапазон пропорций: {self.aspect_ratio_range}")
    
    def _analyze_colors(self, image: np.ndarray, points: List[Tuple[int, int]]):
        """Анализирует цвета документа и фона с улучшенной обработкой и расширенной информацией"""
        # Создаем маску документа
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Расширяем маску немного внутрь, чтобы избежать краевых эффектов
        kernel = np.ones((5, 5), np.uint8)
        mask_inner = cv2.erode(mask, kernel, iterations=3)
        
        # Анализ цвета документа (расширенная информация)
        document_pixels = image[mask_inner == 255]
        if len(document_pixels) > 0:
            # Используем медиану вместо среднего для большей устойчивости к выбросам
            self.avg_color = np.median(document_pixels, axis=0).astype(np.float32)
            # Сохраняем расширенную статистику
            self.document_color_mean = np.mean(document_pixels, axis=0).astype(np.float32)
            self.document_color_std = np.std(document_pixels, axis=0).astype(np.float32)
            self.document_color_min = np.min(document_pixels, axis=0).astype(np.float32)
            self.document_color_max = np.max(document_pixels, axis=0).astype(np.float32)
        else:
            # Fallback на полную маску
            document_pixels = image[mask == 255]
            if len(document_pixels) > 0:
                self.avg_color = np.median(document_pixels, axis=0).astype(np.float32)
                self.document_color_mean = np.mean(document_pixels, axis=0).astype(np.float32)
                self.document_color_std = np.std(document_pixels, axis=0).astype(np.float32)
                self.document_color_min = np.min(document_pixels, axis=0).astype(np.float32)
                self.document_color_max = np.max(document_pixels, axis=0).astype(np.float32)
            else:
                default_color = np.array([128, 128, 128], dtype=np.float32)
                self.avg_color = default_color
                self.document_color_mean = default_color
                self.document_color_std = np.array([10, 10, 10], dtype=np.float32)
                self.document_color_min = default_color
                self.document_color_max = default_color
        
        # Анализ цвета фона (расширенная информация)
        h, w = image.shape[:2]
        bg_mask = cv2.bitwise_not(mask)
        
        # Собираем образцы фона из разных областей ВОКРУГ документа
        self.bg_samples = []
        
        # 1. Образцы из углов изображения (всегда фон)
        corner_regions = [
            image[0:min(100, h//3), 0:min(100, w//3)],  # Верхний левый
            image[0:min(100, h//3), max(0, w-100):w],  # Верхний правый
            image[max(0, h-100):h, 0:min(100, w//3)],  # Нижний левый
            image[max(0, h-100):h, max(0, w-100):w],  # Нижний правый
        ]
        for region in corner_regions:
            if region.size > 0:
                region_flat = region.reshape(-1, 3)
                if len(region_flat) > 0:
                    self.bg_samples.append(np.median(region_flat, axis=0).astype(np.float32))
        
        # 2. Образцы из краев изображения (верх, низ, лево, право)
        edge_regions = [
            image[0:min(50, h//5), :],  # Верх
            image[max(0, h-50):h, :],  # Низ
            image[:, 0:min(50, w//5)],  # Лево
            image[:, max(0, w-50):w],  # Право
        ]
        for region in edge_regions:
            if region.size > 0:
                region_flat = region.reshape(-1, 3)
                if len(region_flat) > 0:
                    self.bg_samples.append(np.median(region_flat, axis=0).astype(np.float32))
        
        # 3. Образцы из области вокруг документа (буферная зона)
        # Расширяем маску документа для создания буферной зоны
        mask_expanded = cv2.dilate(mask, np.ones((50, 50), np.uint8), iterations=3)
        bg_mask_expanded = cv2.bitwise_and(bg_mask, mask_expanded)
        
        # Берем только пиксели достаточно далеко от границы документа
        kernel = np.ones((40, 40), np.uint8)
        bg_mask_clean = cv2.erode(bg_mask_expanded, kernel)
        bg_pixels = image[bg_mask_clean == 255]
        
        # Добавляем образцы из буферной зоны
        if len(bg_pixels) > 50:
            # Берем несколько случайных образцов из буферной зоны
            sample_indices = np.random.choice(len(bg_pixels), min(10, len(bg_pixels)), replace=False)
            for idx in sample_indices:
                self.bg_samples.append(bg_pixels[idx].astype(np.float32))
        
        if len(bg_pixels) > 100:  # Нужно достаточно пикселей для надежной оценки
            self.avg_bg_color = np.median(bg_pixels, axis=0).astype(np.float32)
            self.bg_color_mean = np.mean(bg_pixels, axis=0).astype(np.float32)
            self.bg_color_std = np.std(bg_pixels, axis=0).astype(np.float32)
        else:
            # Fallback: используем края изображения
            edge_pixels = np.concatenate([
                image[0:5, :].reshape(-1, 3),
                image[h-5:h, :].reshape(-1, 3),
                image[:, 0:5].reshape(-1, 3),
                image[:, w-5:w].reshape(-1, 3)
            ])
            if len(edge_pixels) > 0:
                self.avg_bg_color = np.median(edge_pixels, axis=0).astype(np.float32)
                self.bg_color_mean = np.mean(edge_pixels, axis=0).astype(np.float32)
                self.bg_color_std = np.std(edge_pixels, axis=0).astype(np.float32)
            else:
                default_bg = np.array([200, 200, 200], dtype=np.float32)
                self.avg_bg_color = default_bg
                self.bg_color_mean = default_bg
                self.bg_color_std = np.array([10, 10, 10], dtype=np.float32)
        
        # Вычисляем порог по цвету в LAB пространстве (более точное)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_color_lab = cv2.cvtColor(
            np.uint8([[self.avg_color]]), 
            cv2.COLOR_BGR2LAB
        )[0][0]
        avg_bg_color_lab = cv2.cvtColor(
            np.uint8([[self.avg_bg_color]]), 
            cv2.COLOR_BGR2LAB
        )[0][0]
        
        color_diff = np.linalg.norm(avg_color_lab.astype(np.float32) - avg_bg_color_lab.astype(np.float32))
        # Более консервативный порог
        self.color_threshold = max(25, min(60, color_diff * 0.4))
    
    def _analyze_edges(self, image: np.ndarray, points: List[Tuple[int, int]]):
        """Анализирует края документа с улучшенной обработкой"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Предобработка: уменьшаем шум
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Адаптивный выбор порогов Canny
        median = np.median(gray)
        low_threshold = int(max(0, 0.7 * median))
        high_threshold = int(min(255, 1.3 * median))
        
        # Детектор краев на калибровочном изображении
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Создаем маску документа
        mask = np.zeros_like(edges)
        points_array = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Расширяем маску немного внутрь для анализа краев внутри документа
        kernel = np.ones((3, 3), np.uint8)
        mask_inner = cv2.erode(mask, kernel, iterations=2)
        
        # Анализируем силу краев внутри документа (но не на границе)
        document_edges = edges & mask_inner
        edge_pixels = document_edges[document_edges > 0]
        
        if len(edge_pixels) > 0:
            # Используем медиану для большей устойчивости
            edge_strength = np.median(edge_pixels)
        else:
            # Fallback: анализируем края на границе документа
            mask_border = cv2.bitwise_xor(mask, mask_inner)
            border_edges = edges & mask_border
            edge_pixels = border_edges[border_edges > 0]
            edge_strength = np.median(edge_pixels) if len(edge_pixels) > 0 else 50
        
        # Устанавливаем порог на основе силы краев
        # Используем более консервативный подход
        self.edge_threshold = int(max(40, min(150, edge_strength * 0.8)))
    
    def _analyze_geometry(self, points: List[Tuple[int, int]], image_size: Tuple[int, int]):
        """Анализирует геометрические параметры с улучшенной обработкой и расширенной информацией"""
        w, h = image_size
        points_array = np.array(points, dtype=np.float32)
        
        # Вычисляем площадь документа
        area = cv2.contourArea(points_array)
        area_ratio = area / (w * h)
        
        # Сохраняем площадь в пикселях и как долю
        self.document_area_pixels = int(area)
        self.document_area_ratio = area_ratio
        
        # Вычисляем соотношение сторон через минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(points_array)
        width, height = rect[1]
        
        # Убеждаемся что width и height положительные
        if width < height:
            width, height = height, width
        
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Сохраняем размеры и соотношение сторон
        self.document_aspect_ratio = aspect_ratio
        
        # Вычисляем размеры через перспективное преобразование для точности
        ordered_pts = self._order_points_for_size(points_array)
        width_calc = max(
            np.linalg.norm(ordered_pts[1] - ordered_pts[0]),
            np.linalg.norm(ordered_pts[2] - ordered_pts[3])
        )
        height_calc = max(
            np.linalg.norm(ordered_pts[3] - ordered_pts[0]),
            np.linalg.norm(ordered_pts[2] - ordered_pts[1])
        )
        
        # Сохраняем размеры документа
        self.document_width = int(width_calc)
        self.document_height = int(height_calc)
        self.target_size = (self.document_width, self.document_height)
        
        # Устанавливаем диапазоны с более широким допуском для большей гибкости
        # Площадь: от 50% до 150% от калибровочной
        area_min = max(0.05, area_ratio * 0.5)  # Минимум 5% изображения
        area_max = min(0.95, area_ratio * 1.5)  # Максимум 95% изображения
        self.area_range = (area_min, area_max)
        
        # Соотношение сторон: от 70% до 130% от калибровочного
        aspect_min = max(1.0, aspect_ratio * 0.7)
        aspect_max = min(10.0, aspect_ratio * 1.3)
        self.aspect_ratio_range = (aspect_min, aspect_max)
    
    def _order_points_for_size(self, pts: np.ndarray) -> np.ndarray:
        """Вспомогательная функция для упорядочивания точек"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

class CalibrationManager:
    """Менеджер калибровки для настройки параметров обрезки"""
    
    def __init__(self, calibration_config: CalibrationConfig):
        self.config = calibration_config
        self.current_points: List[Tuple[int, int]] = []
        self.current_image: Optional[np.ndarray] = None
        self.image_paths: List[str] = []
        self.current_index = 0
        
    def load_images_from_folder(self, folder_path: str) -> bool:
        """Загружает изображения из папки для калибровки"""
        folder = Path(folder_path)
        if not folder.exists():
            return False
            
        # Ищем все JPEG файлы
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend([str(p) for p in folder.glob(ext)])
        
        if not self.image_paths:
            return False
            
        # Перемешиваем для случайного порядка
        random.shuffle(self.image_paths)
        self.current_index = 0
        
        print(f"📁 Загружено {len(self.image_paths)} изображений для калибровки")
        return True
    
    def get_next_calibration_image(self) -> Optional[Tuple[np.ndarray, str]]:
        """Возвращает следующее изображение для калибровки"""
        if self.current_index >= len(self.image_paths):
            return None
            
        image_path = self.image_paths[self.current_index]
        image = cv2.imread(image_path)
        
        if image is None:
            return None
            
        self.current_image = image
        self.current_points = []
        
        self.current_index += 1
        return image, Path(image_path).name
    
    def add_point(self, x: int, y: int) -> bool:
        """Добавляет точку для калибровки"""
        if self.current_image is None:
            return False
            
        if len(self.current_points) >= 4:
            return False
            
        self.current_points.append((x, y))
        return True
    
    def remove_last_point(self):
        """Удаляет последнюю добавленную точку"""
        if self.current_points:
            self.current_points.pop()
    
    def clear_points(self):
        """Очищает все точки"""
        self.current_points = []
    
    def save_calibration(self) -> bool:
        """Сохраняет текущую калибровку"""
        if self.current_image is None or len(self.current_points) != 4:
            return False
            
        # Анализируем изображение и извлекаем параметры
        self.config.analyze_calibration_image(self.current_image, self.current_points)
        return True
    
    def get_annotated_image(self) -> Optional[np.ndarray]:
        """Возвращает изображение с отмеченными точками и контуром"""
        if self.current_image is None:
            return None
            
        image = self.current_image.copy()
        
        # Рисуем контур если есть 4 точки
        if len(self.current_points) == 4:
            points = np.array(self.current_points, dtype=np.int32)
            cv2.polylines(image, [points], True, (0, 255, 0), 3)
            
            # Заливаем область прозрачным цветом
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # Рисуем точки
        for i, (x, y) in enumerate(self.current_points):
            color = (0, 0, 255) if i < 4 else (255, 0, 0)
            cv2.circle(image, (x, y), 10, color, -1)
            cv2.putText(image, str(i + 1), (x + 15, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Добавляем инструкцию
        instruction = "Щелкните 4 угла документа (по часовой стрелке)"
        cv2.putText(image, instruction, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(image, f"Точек: {len(self.current_points)}/4", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def get_progress(self) -> Tuple[int, int]:
        """Возвращает прогресс калибровки"""
        return self.current_index, len(self.image_paths)
    
    def is_complete(self) -> bool:
        """Проверяет завершена ли калибровка"""
        return self.config.calibrated and self.config.calibration_samples >= 1
