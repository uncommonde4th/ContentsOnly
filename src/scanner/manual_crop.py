import cv2
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from scanner.calibration import CalibrationConfig
from scanner.image_processor import CalibratedImageProcessor

class ManualCropConfig:
    """Конфигурация для ручной обрезки с обучением на основе предыдущих обрезок"""
    def __init__(self):
        self.crop_points_history: List[List[Tuple[float, float]]] = []  # История точек в процентах
        self.image_sizes_history: List[Tuple[int, int]] = []  # История размеров изображений
        self.samples_count = 0
        
    def add_sample(self, points: List[Tuple[int, int]], image_size: Tuple[int, int]):
        """Добавляет образец обрезки для обучения"""
        w, h = image_size
        # Сохраняем точки в процентах
        points_normalized = [(x / w, y / h) for x, y in points]
        self.crop_points_history.append(points_normalized)
        self.image_sizes_history.append(image_size)
        self.samples_count += 1
        
    def get_suggested_points(self, image_size: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Получает предложенные точки на основе истории"""
        if self.samples_count == 0:
            return None
            
        w, h = image_size
        
        # Вычисляем средние точки из истории
        if len(self.crop_points_history) > 0:
            # Берем последние несколько образцов (до 10) для более точных предсказаний
            recent_samples = self.crop_points_history[-min(10, len(self.crop_points_history)):]
            
            # Вычисляем средние координаты
            avg_points = []
            for i in range(4):
                x_coords = [sample[i][0] for sample in recent_samples]
                y_coords = [sample[i][1] for sample in recent_samples]
                
                # Используем медиану для большей устойчивости к выбросам
                avg_x = np.median(x_coords)
                avg_y = np.median(y_coords)
                
                avg_points.append((int(avg_x * w), int(avg_y * h)))
            
            return avg_points
        
        return None

class ManualCropManager:
    """Менеджер для ручной обрезки с интерактивными точками"""
    
    def __init__(self, manual_crop_config: ManualCropConfig, calibration_config: Optional[CalibrationConfig] = None):
        self.config = manual_crop_config
        self.calibration_config = calibration_config
        self.current_points: List[Tuple[int, int]] = []
        self.current_image: Optional[np.ndarray] = None
        self.image_paths: List[str] = []
        self.current_index = 0
        self.saved_indices: set = set()  # Индексы сохраненных изображений
        self.skipped_indices: set = set()  # Индексы пропущенных изображений
        self.dragging_point_index: Optional[int] = None
        self.hover_point_index: Optional[int] = None
        self.dragging_area: bool = False
        self.drag_start_offset: Optional[Tuple[int, int]] = None  # Смещение от точки клика до центра области
        self.dragging_edge: Optional[int] = None  # Индекс стороны для перетаскивания (0-3)
        self.drag_edge_offset: Optional[Tuple[int, int]] = None  # Перпендикулярное смещение для перетаскивания стороны
        self.drag_start_mouse_pos: Optional[Tuple[int, int]] = None  # Начальная позиция мыши при перетаскивании стороны
        
    def load_images_from_folder(self, folder_path: str) -> bool:
        """Загружает изображения из папки для ручной обрезки"""
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
            
        # Сортируем для последовательной обработки
        self.image_paths.sort()
        self.current_index = 0
        self.saved_indices = set()
        self.skipped_indices = set()
        
        print(f"📁 Загружено {len(self.image_paths)} изображений для ручной обрезки")
        return True
    
    def get_next_image(self) -> Optional[Tuple[np.ndarray, str]]:
        """Возвращает следующее изображение для обрезки"""
        # Сначала ищем непропущенные изображения
        while self.current_index < len(self.image_paths):
            if self.current_index not in self.saved_indices and self.current_index not in self.skipped_indices:
                break
            self.current_index += 1
        
        # Если все обработаны, показываем пропущенные
        if self.current_index >= len(self.image_paths):
            if self.skipped_indices:
                # Показываем пропущенные по порядку
                skipped_list = sorted(list(self.skipped_indices))
                if skipped_list:
                    self.current_index = skipped_list[0]
                    self.skipped_indices.remove(self.current_index)
                else:
                    return None
            else:
                return None
        
        image_path = self.image_paths[self.current_index]
        image = cv2.imread(image_path)
        
        if image is None:
            self.current_index += 1
            return self.get_next_image()  # Пробуем следующее
            
        self.current_image = image
        self.current_points = []
        self.dragging_point_index = None
        self.hover_point_index = None
        self.dragging_area = False
        self.drag_start_offset = None
        self.dragging_edge = None
        self.drag_edge_offset = None
        self.drag_start_mouse_pos = None
        
        # НЕ увеличиваем current_index здесь - это будет сделано при сохранении или пропуске
        return image, Path(image_path).name
    
    def skip_current_image(self):
        """Пропускает текущее изображение без сохранения"""
        if self.current_index < len(self.image_paths):
            self.skipped_indices.add(self.current_index)
            self.current_index += 1
    
    def get_previous_image(self) -> Optional[Tuple[np.ndarray, str]]:
        """Возвращает предыдущее изображение"""
        # Находим предыдущее несохраненное изображение
        if self.current_index <= 0:
            return None
        
        # Ищем предыдущее изображение которое не сохранено
        prev_index = self.current_index - 1
        while prev_index >= 0:
            if prev_index not in self.saved_indices:
                self.current_index = prev_index
                image_path = self.image_paths[self.current_index]
                image = cv2.imread(image_path)
                
                if image is None:
                    prev_index -= 1
                    continue
                
                self.current_image = image
                self.current_points = []
                self.dragging_point_index = None
                self.hover_point_index = None
                self.dragging_area = False
                self.drag_start_offset = None
                self.dragging_edge = None
                self.drag_edge_offset = None
                self.drag_start_mouse_pos = None
                
                return image, Path(image_path).name
            prev_index -= 1
        
        return None
    
    def add_point(self, x: int, y: int) -> bool:
        """Добавляет точку для обрезки"""
        if self.current_image is None:
            return False
            
        if len(self.current_points) >= 4:
            return False
            
        self.current_points.append((x, y))
        return True
    
    def set_points(self, points: List[Tuple[int, int]]):
        """Устанавливает все 4 точки сразу"""
        if len(points) == 4:
            self.current_points = points.copy()
    
    def remove_last_point(self):
        """Удаляет последнюю добавленную точку"""
        if self.current_points:
            self.current_points.pop()
    
    def clear_points(self):
        """Очищает все точки"""
        self.current_points = []
        self.dragging_point_index = None
        self.hover_point_index = None
        self.dragging_area = False
        self.drag_start_offset = None
        self.dragging_edge = None
        self.drag_edge_offset = None
        self.drag_start_mouse_pos = None
    
    def is_point_inside_area(self, x: int, y: int) -> bool:
        """Проверяет находится ли точка внутри выделенной области"""
        if len(self.current_points) != 4:
            return False
        
        # Используем алгоритм ray casting для проверки точки внутри многоугольника
        points = np.array(self.current_points, dtype=np.int32)
        return cv2.pointPolygonTest(points, (x, y), False) >= 0
    
    def find_nearest_edge(self, x: int, y: int, threshold: int = 20) -> Optional[int]:
        """Находит ближайшую сторону области"""
        if len(self.current_points) != 4:
            return None
        
        min_distance = float('inf')
        nearest_edge = None
        
        for i in range(4):
            p1 = self.current_points[i]
            p2 = self.current_points[(i + 1) % 4]
            
            # Вычисляем расстояние от точки до отрезка
            # Используем формулу расстояния от точки до отрезка
            A = np.array([p1[0], p1[1]], dtype=np.float32)
            B = np.array([p2[0], p2[1]], dtype=np.float32)
            P = np.array([x, y], dtype=np.float32)
            
            # Вектор AB
            AB = B - A
            # Вектор AP
            AP = P - A
            
            # Проекция AP на AB
            ab_sq = np.dot(AB, AB)
            if ab_sq == 0:
                continue
            
            t = np.clip(np.dot(AP, AB) / ab_sq, 0.0, 1.0)
            
            # Ближайшая точка на отрезке
            closest = A + t * AB
            
            # Расстояние от P до ближайшей точки на отрезке
            distance = np.linalg.norm(P - closest)
            
            if distance < min_distance and distance <= threshold:
                min_distance = distance
                nearest_edge = i
        
        return nearest_edge
    
    def start_edge_drag(self, x: int, y: int) -> bool:
        """Начинает перетаскивание стороны"""
        if len(self.current_points) != 4:
            return False
        
        edge_idx = self.find_nearest_edge(x, y)
        if edge_idx is not None:
            # Вычисляем смещение от точки клика до ближайшей точки на стороне
            p1 = self.current_points[edge_idx]
            p2 = self.current_points[(edge_idx + 1) % 4]
            
            A = np.array([p1[0], p1[1]], dtype=np.float32)
            B = np.array([p2[0], p2[1]], dtype=np.float32)
            P = np.array([x, y], dtype=np.float32)
            
            AB = B - A
            AP = P - A
            
            ab_sq = np.dot(AB, AB)
            if ab_sq > 0:
                t = np.clip(np.dot(AP, AB) / ab_sq, 0.0, 1.0)
                closest = A + t * AB
                # Сохраняем расстояние от точки клика до стороны (перпендикулярное расстояние)
                perp_vector = P - closest
                self.drag_edge_offset = (int(perp_vector[0]), int(perp_vector[1]))
            else:
                self.drag_edge_offset = (0, 0)
            
            # Сохраняем начальную позицию мыши для отслеживания смещения
            self.drag_start_mouse_pos = (x, y)
            self.dragging_edge = edge_idx
            return True
        
        return False
    
    def start_area_drag(self, x: int, y: int) -> bool:
        """Начинает перетаскивание всей области"""
        if len(self.current_points) != 4:
            return False
        
        if not self.is_point_inside_area(x, y):
            return False
        
        # Вычисляем центр области
        points_array = np.array(self.current_points, dtype=np.float32)
        center = np.mean(points_array, axis=0)
        
        # Сохраняем смещение от точки клика до центра
        self.drag_start_offset = (int(x - center[0]), int(y - center[1]))
        self.dragging_area = True
        return True
    
    def find_point_at(self, x: int, y: int, threshold: int = 40) -> Optional[int]:
        """Находит точку рядом с указанными координатами"""
        for i, (px, py) in enumerate(self.current_points):
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance <= threshold:
                return i
        return None
    
    def start_drag(self, x: int, y: int) -> bool:
        """Начинает перетаскивание точки, стороны или области"""
        # Сначала проверяем перетаскивание точки
        point_idx = self.find_point_at(x, y)
        if point_idx is not None:
            self.dragging_point_index = point_idx
            return True
        
        # Затем проверяем перетаскивание стороны
        if len(self.current_points) == 4 and self.start_edge_drag(x, y):
            return True
        
        # Если не попали в точку или сторону, проверяем перетаскивание области
        if self.start_area_drag(x, y):
            return True
        
        return False
    
    def update_drag(self, x: int, y: int):
        """Обновляет позицию перетаскиваемой точки, стороны или области"""
        if self.current_image is None:
            return
        
        h, w = self.current_image.shape[:2]
        
        if self.dragging_point_index is not None:
            # Перетаскивание отдельной точки
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            self.current_points[self.dragging_point_index] = (x, y)
        elif self.dragging_edge is not None and self.drag_start_mouse_pos is not None:
            # Перетаскивание стороны - двигаем перпендикулярно направлению стороны
            edge_idx = self.dragging_edge
            p1 = self.current_points[edge_idx]
            p2 = self.current_points[(edge_idx + 1) % 4]
            
            # Вычисляем направление стороны
            A = np.array([p1[0], p1[1]], dtype=np.float32)
            B = np.array([p2[0], p2[1]], dtype=np.float32)
            AB = B - A
            ab_len = np.linalg.norm(AB)
            
            if ab_len > 0:
                # Нормализованный вектор стороны
                AB_norm = AB / ab_len
                
                # Перпендикулярный вектор (поворот на 90 градусов)
                perp_norm = np.array([-AB_norm[1], AB_norm[0]], dtype=np.float32)
                
                # Вычисляем текущую позицию курсора
                current_mouse = np.array([x, y], dtype=np.float32)
                
                # Находим ближайшую точку на стороне к начальной позиции мыши
                start_x, start_y = self.drag_start_mouse_pos
                start_mouse = np.array([start_x, start_y], dtype=np.float32)
                
                # Проекция начальной позиции мыши на сторону
                AP_start = start_mouse - A
                t_start = np.clip(np.dot(AP_start, AB_norm) / ab_len, 0.0, 1.0)
                closest_start = A + t_start * AB
                
                # Перпендикулярное расстояние от начальной позиции мыши до стороны
                perp_dist_start = np.dot(start_mouse - closest_start, perp_norm)
                
                # Проекция текущей позиции мыши на сторону
                AP_current = current_mouse - A
                t_current = np.clip(np.dot(AP_current, AB_norm) / ab_len, 0.0, 1.0)
                closest_current = A + t_current * AB
                
                # Перпендикулярное расстояние от текущей позиции мыши до стороны
                perp_dist_current = np.dot(current_mouse - closest_current, perp_norm)
                
                # Вычисляем смещение стороны (разница перпендикулярных расстояний)
                perp_displacement = perp_dist_current - perp_dist_start
                
                # Применяем смещение к обеим точкам стороны в перпендикулярном направлении
                displacement_vector = perp_norm * perp_displacement
                
                new_p1 = (int(p1[0] + displacement_vector[0]), int(p1[1] + displacement_vector[1]))
                new_p2 = (int(p2[0] + displacement_vector[0]), int(p2[1] + displacement_vector[1]))
                
                # Ограничиваем координаты
                new_p1 = (max(0, min(new_p1[0], w - 1)), max(0, min(new_p1[1], h - 1)))
                new_p2 = (max(0, min(new_p2[0], w - 1)), max(0, min(new_p2[1], h - 1)))
                
                # Обновляем точки
                new_points = list(self.current_points)
                new_points[edge_idx] = new_p1
                new_points[(edge_idx + 1) % 4] = new_p2
                self.current_points = new_points
                
                # Обновляем начальную позицию мыши для следующего кадра
                self.drag_start_mouse_pos = (x, y)
        elif self.dragging_area and self.drag_start_offset is not None:
            # Перетаскивание всей области
            # Вычисляем новый центр области
            new_center_x = x - self.drag_start_offset[0]
            new_center_y = y - self.drag_start_offset[1]
            
            # Вычисляем текущий центр
            points_array = np.array(self.current_points, dtype=np.float32)
            current_center = np.mean(points_array, axis=0)
            
            # Вычисляем смещение
            dx = new_center_x - current_center[0]
            dy = new_center_y - current_center[1]
            
            # Применяем смещение ко всем точкам
            new_points = []
            for px, py in self.current_points:
                new_x = int(px + dx)
                new_y = int(py + dy)
                
                # Ограничиваем координаты пределами изображения
                new_x = max(0, min(new_x, w - 1))
                new_y = max(0, min(new_y, h - 1))
                
                new_points.append((new_x, new_y))
            
            self.current_points = new_points
    
    def end_drag(self):
        """Заканчивает перетаскивание"""
        self.dragging_point_index = None
        self.dragging_area = False
        self.drag_start_offset = None
        self.dragging_edge = None
        self.drag_edge_offset = None
    
    def update_hover(self, x: int, y: int):
        """Обновляет индекс точки под курсором"""
        self.hover_point_index = self.find_point_at(x, y)
    
    def get_annotated_image(self) -> Optional[np.ndarray]:
        """Возвращает изображение с отмеченными точками и контуром"""
        if self.current_image is None:
            return None
            
        image = self.current_image.copy()
        
        # Рисуем контур если есть 4 точки
        if len(self.current_points) == 4:
            points = np.array(self.current_points, dtype=np.int32)
            
            # Если перетаскиваем область или сторону, используем другой цвет
            if self.dragging_area or self.dragging_edge is not None:
                cv2.polylines(image, [points], True, (255, 255, 0), 4)  # Желтый при перетаскивании
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], (255, 255, 0))
                cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
            else:
                cv2.polylines(image, [points], True, (0, 255, 0), 3)  # Зеленый обычно
                overlay = image.copy()
                cv2.fillPoly(overlay, [points], (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, image, 0.8, 0, image)
        
        # Рисуем точки с улучшенным визуальным отображением (увеличенные размеры)
        for i, (x, y) in enumerate(self.current_points):
            # Определяем цвет и размер точки (увеличено еще больше)
            if self.dragging_point_index == i:
                color = (255, 255, 0)  # Желтый для перетаскиваемой
                radius = 40
                thickness = 5
            elif self.hover_point_index == i:
                color = (0, 255, 255)  # Голубой для наведения
                radius = 32
                thickness = 4
            else:
                color = (0, 0, 255)  # Красный для обычных
                radius = 28
                thickness = 4
            
            # Рисуем внешний круг (белый фон)
            cv2.circle(image, (x, y), radius + 6, (255, 255, 255), -1)
            # Рисуем основной круг
            cv2.circle(image, (x, y), radius, color, thickness)
            # Рисуем внутренний круг
            cv2.circle(image, (x, y), radius - 8, color, -1)
            
            # Номер точки (увеличенный шрифт)
            font_scale = 1.5
            cv2.putText(image, str(i + 1), (x + radius + 15, y - radius - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4)
            cv2.putText(image, str(i + 1), (x + radius + 15, y - radius - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        
        # Добавляем инструкцию
        if len(self.current_points) < 4:
            instruction = f"Щелкните 4 угла документа (по часовой стрелке) - {len(self.current_points)}/4"
        else:
            if self.dragging_area:
                instruction = "Перетаскивание области... Отпустите кнопку мыши для завершения"
            elif self.dragging_edge is not None:
                instruction = "Перетаскивание стороны... Отпустите кнопку мыши для завершения"
            else:
                instruction = "Все точки отмечены! Перетащите точки, стороны или область для корректировки, затем сохраните результат"
        
        cv2.putText(image, instruction, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image, instruction, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        return image
    
    def crop_image(self) -> Optional[np.ndarray]:
        """Обрезает изображение по текущим точкам"""
        if self.current_image is None or len(self.current_points) != 4:
            return None
        
        # Используем функцию из image_processor для перспективного преобразования
        processor = CalibratedImageProcessor(None, self.calibration_config) if self.calibration_config else None
        
        if processor:
            points_array = np.array(self.current_points, dtype=np.float32)
            result = processor.four_point_transform(self.current_image, points_array)
        else:
            # Простое перспективное преобразование без калибровки
            result = self._simple_four_point_transform(self.current_image, self.current_points)
        
        return result
    
    def _simple_four_point_transform(self, image: np.ndarray, pts: List[Tuple[int, int]]) -> np.ndarray:
        """Простое перспективное преобразование по 4 точкам"""
        pts_array = np.array(pts, dtype=np.float32)
        
        # Упорядочиваем точки
        rect = self._order_points(pts_array)
        (tl, tr, br, bl) = rect
        
        # Вычисляем размеры
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Формируем точки назначения
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Вычисляем матрицу преобразования
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Применяем преобразование
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Упорядочивает точки: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect
    
    def save_crop(self, output_path: str) -> bool:
        """Сохраняет обрезанное изображение и добавляет в калибровку"""
        if self.current_image is None or len(self.current_points) != 4:
            return False
        
        # Обрезаем изображение
        cropped = self.crop_image()
        if cropped is None:
            return False
        
        # Сохраняем
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        # Добавляем в калибровку для улучшения алгоритма
        if self.calibration_config is not None:
            print("🔧 Добавляем в калибровку для улучшения алгоритма...")
            self.calibration_config.analyze_calibration_image(self.current_image, self.current_points)
            print(f"   ✅ Калибровка обновлена (образцов: {self.calibration_config.calibration_samples})")
        
        # Добавляем образец для истории ручной обрезки
        h, w = self.current_image.shape[:2]
        self.config.add_sample(self.current_points, (w, h))
        
        # Помечаем как сохраненное
        if self.current_index < len(self.image_paths):
            self.saved_indices.add(self.current_index)
            self.current_index += 1
        
        return True
    
    def get_suggested_points(self) -> Optional[List[Tuple[int, int]]]:
        """Получает предложенные точки используя алгоритм из автоматической обработки"""
        if self.current_image is None:
            return None
        
        h, w = self.current_image.shape[:2]
        
        print("🔍 Анализ изображения для подсказки обрезки...")
        
        # Всегда используем алгоритм из автоматической обработки
        try:
            from scanner.image_processor import ProcessingConfig
            processing_config = ProcessingConfig()
            
            # Создаем процессор с калибровкой (если есть) или без неё
            processor = CalibratedImageProcessor(processing_config, self.calibration_config)
            
            contour = None
            
            # Если калибровка есть - используем полный алгоритм
            if self.calibration_config and self.calibration_config.calibrated:
                print("   Используем алгоритм автоматической обработки с калибровкой...")
                contour = processor.find_document_auto(self.current_image)
            
            # Если не нашли или калибровки нет - пробуем все методы вручную
            if contour is None:
                print("   Пробуем все методы алгоритма автоматической обработки...")
                
                # Метод 0: Светлый документ на темном фоне
                if self.calibration_config and self.calibration_config.calibrated:
                    contour = processor._find_light_on_dark(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден как светлый документ на темном фоне")
                
                # Метод 0.5: Края документа
                if contour is None and self.calibration_config and self.calibration_config.calibrated:
                    contour = processor._find_document_edges(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден по краям документа")
                
                # Метод 1: Поиск по краям
                if contour is None and self.calibration_config and self.calibration_config.calibrated:
                    contour = processor._find_by_edges(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден по краям")
                
                # Метод 2: Поиск по цвету
                if contour is None and self.calibration_config and self.calibration_config.calibrated:
                    contour = processor._find_by_color(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден по цвету")
                
                # Метод 3: Поиск по текстуре
                if contour is None and self.calibration_config and self.calibration_config.calibrated:
                    contour = processor._find_by_texture(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден по текстуре")
                
                # Метод 4: Ослабленные ограничения
                if contour is None:
                    contour = processor._find_with_relaxed_constraints(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден с ослабленными ограничениями")
                
                # Метод 5: Любой большой прямоугольник (работает без калибровки)
                if contour is None:
                    contour = processor._find_any_large_rectangle(self.current_image)
                    if contour is not None:
                        print("   ✅ Найден большой прямоугольный контур")
            
            if contour is not None:
                # Преобразуем контур в список точек
                points = contour.reshape(4, 2).tolist()
                detected_points = [(int(p[0]), int(p[1])) for p in points]
                print(f"   ✅ Документ найден алгоритмом автоматической обработки: {detected_points}")
                return detected_points
            else:
                print("   ⚠️ Алгоритм автоматической обработки не нашел документ")
                
        except Exception as e:
            print(f"   ⚠️ Ошибка при использовании алгоритма автоматической обработки: {e}")
            import traceback
            traceback.print_exc()
        
        # Если ничего не нашли, пробуем использовать только историю как fallback
        print("   Используем историю как fallback...")
        historical_points = self.config.get_suggested_points((w, h))
        if historical_points:
            print(f"   ✅ Используем точки из истории: {historical_points}")
        else:
            print("   ❌ Подсказка недоступна")
        return historical_points
    
    def _validate_quadrilateral(self, points: List[Tuple[int, int]], image_area: float) -> bool:
        """Проверяет что четырехугольник валиден и похож на документ"""
        if len(points) != 4:
            return False
        
        pts_array = np.array(points, dtype=np.float32)
        
        # Проверяем площадь
        area = cv2.contourArea(pts_array)
        if area < image_area * 0.1 or area > image_area * 0.95:  # От 10% до 95%
            return False
        
        # Проверяем прямоугольность через минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(pts_array)
        width, height = rect[1]
        if min(width, height) < 50:  # Минимальный размер
            return False
        
        # Соотношение сторон должно быть разумным (не слишком вытянутое)
        aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        if aspect_ratio > 10.0:  # Слишком вытянутое
            return False
        
        # Проверяем выпуклость
        hull = cv2.convexHull(pts_array)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.85:  # Должно быть достаточно выпуклым
                return False
        
        # Проверяем углы (должны быть близки к 90 градусам)
        angles = []
        for i in range(4):
            p1 = pts_array[i]
            p2 = pts_array[(i + 1) % 4]
            p3 = pts_array[(i + 2) % 4]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        
        # Среднее отклонение от 90 градусов не должно быть слишком большим
        avg_deviation = np.mean([abs(a - 90) for a in angles])
        if avg_deviation > 45:  # Слишком не прямоугольный
            return False
        
        return True
    
    def _get_best_corners(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """Находит 4 лучших угловых точки из контура"""
        # Сначала пробуем упростить контур
        for eps_factor in [0.01, 0.02, 0.03, 0.05, 0.08]:
            epsilon = eps_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return approx
            elif len(approx) > 4:
                # Пробуем найти 4 угловые точки
                hull = cv2.convexHull(contour)
                if len(hull) >= 4:
                    hull_pts = hull.reshape(-1, 2).astype(np.float32)
                    
                    # Находим центр
                    center = np.mean(hull_pts, axis=0)
                    
                    # Вычисляем углы от центра
                    angles = []
                    for pt in hull_pts:
                        angle = np.arctan2(pt[1] - center[1], pt[0] - center[0])
                        angles.append((angle, pt))
                    angles.sort()
                    
                    # Берем 4 точки равномерно распределенные по углам
                    if len(angles) >= 4:
                        step = len(angles) // 4
                        selected = [angles[i * step][1] for i in range(4)]
                        return np.array(selected, dtype=np.int32).reshape(-1, 1, 2)
        
        return None
    
    
    def get_progress(self) -> Tuple[int, int]:
        """Возвращает прогресс обработки"""
        processed = len(self.saved_indices)
        total = len(self.image_paths)
        return processed, total
    
    def has_more_images(self) -> bool:
        """Проверяет есть ли еще изображения"""
        return self.current_index < len(self.image_paths)

