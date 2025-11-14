import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

class ProcessingConfig:
    """Простая конфигурация обработки"""
    def __init__(self):
        self.jpeg_quality = 95

class CalibratedImageProcessor:
    """Обработчик изображений с использованием параметров калибровки"""
    
    def __init__(self, processing_config: ProcessingConfig, calibration_config):
        self.processing_config = processing_config
        self.calibration_config = calibration_config
    
    def find_document_auto(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Автоматически находит документ на изображении используя параметры калибровки"""
        if not self.calibration_config.calibrated:
            return None
        
        print("🔍 Поиск документа с параметрами калибровки...")
        
        # Метод 1: Поиск по краям (самый надежный)
        contour = self._find_by_edges(image)
        if contour is not None:
            print("✅ Найден по краям")
            return contour
        
        # Метод 2: Поиск по цвету (LAB цветовое пространство)
        contour = self._find_by_color(image)
        if contour is not None:
            print("✅ Найден по цвету")
            return contour
        
        # Метод 3: Поиск по текстурам
        contour = self._find_by_texture(image)
        if contour is not None:
            print("✅ Найден по текстуре")
            return contour
        
        print("❌ Документ не найден автоматически")
        return None
    
    def _find_by_color(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Поиск документа по цвету используя LAB цветовое пространство (более точное)"""
        if self.calibration_config.avg_color is None:
            return None
        
        # Конвертируем в LAB цветовое пространство (более восприимчиво к изменениям освещения)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_color_lab = cv2.cvtColor(
            np.uint8([[self.calibration_config.avg_color]]), 
            cv2.COLOR_BGR2LAB
        )[0][0]
        
        # Вычисляем разницу в LAB пространстве (более точная метрика)
        color_diff = np.linalg.norm(
            image_lab.astype(np.float32) - avg_color_lab.astype(np.float32), 
            axis=2
        )
        
        # Адаптивный порог
        threshold = max(20, min(80, self.calibration_config.color_threshold * 1.5))
        
        # Бинаризация
        _, binary = cv2.threshold(
            color_diff.astype(np.uint8), 
            int(threshold), 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Улучшенные морфологические операции
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        
        # Удаляем шум
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
        # Заполняем пробелы
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        
        return self._find_best_contour(binary, image.shape)
    
    def _find_by_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Улучшенный поиск документа по краям"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Предобработка: уменьшаем шум
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Адаптивные параметры Canny на основе калибровки
        if self.calibration_config.edge_threshold > 0:
            low_threshold = max(30, int(self.calibration_config.edge_threshold * 0.5))
            high_threshold = min(200, int(self.calibration_config.edge_threshold * 1.5))
        else:
            # Автоматический выбор порогов
            median = np.median(gray)
            low_threshold = int(max(0, 0.7 * median))
            high_threshold = int(min(255, 1.3 * median))
        
        # Детектор краев
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Улучшаем края: соединяем близкие линии
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return self._find_best_contour(edges, image.shape)
    
    def _find_by_texture(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Поиск документа по текстуре (для текстовых документов)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Улучшенная предобработка
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Адаптивный порог для текста с большим размером блока
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            15, 5
        )
        
        # Морфологические операции для текстовых областей
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return self._find_best_contour(binary, image.shape)
    
    def _find_best_contour(self, binary: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Находит лучший контур удовлетворяющий параметрам калибровки"""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Сортируем контуры по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        h, w = image_shape[:2]
        image_area = w * h
        best_contour = None
        best_score = 0
        
        # Проверяем до 10 крупнейших контуров
        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            
            # Минимальная площадь (хотя бы 5% изображения)
            if area < image_area * 0.05:
                continue
            
            # Аппроксимируем контур с более точным epsilon
            epsilon = 0.015 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Нужно 4 точки для прямоугольника
            if len(approx) < 4:
                continue
            
            # Если больше 4 точек, пытаемся упростить
            if len(approx) > 4:
                epsilon = 0.03 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) != 4:
                    continue
            
            # Проверяем выпуклость
            if not cv2.isContourConvex(approx):
                continue
            
            area_ratio = area / image_area
            
            # Проверяем площадь (с более мягкими границами если диапазон слишком узкий)
            area_min, area_max = self.calibration_config.area_range
            if area_min > 0 and area_max > 0:
                # Расширяем диапазон на 20% для большей гибкости
                area_min = max(0.05, area_min * 0.8)
                area_max = min(0.95, area_max * 1.2)
                if not (area_min <= area_ratio <= area_max):
                    continue
            
            # Проверяем соотношение сторон
            rect = cv2.minAreaRect(approx)
            width, height = rect[1]
            if min(width, height) < 10:  # Слишком маленький
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            
            # Проверяем соотношение сторон (с более мягкими границами)
            aspect_min, aspect_max = self.calibration_config.aspect_ratio_range
            if aspect_min > 0 and aspect_max > 0:
                aspect_min = max(1.0, aspect_min * 0.7)
                aspect_max = min(10.0, aspect_max * 1.3)
                if not (aspect_min <= aspect_ratio <= aspect_max):
                    continue
            
            # Оцениваем контур
            score = self._score_contour(approx, area_ratio, aspect_ratio, image_area)
            if score > best_score:
                best_score = score
                best_contour = approx
        
        return best_contour
    
    def _score_contour(self, contour: np.ndarray, area_ratio: float, aspect_ratio: float, image_area: float) -> float:
        """Оценивает качество контура"""
        score = 0.0
        
        # Оценка по площади (ближе к середине диапазона - лучше)
        area_min, area_max = self.calibration_config.area_range
        if area_min > 0 and area_max > 0:
            target_area = (area_min + area_max) / 2
            area_diff = abs(area_ratio - target_area) / max(target_area, 0.01)
            score += max(0, 1.0 - area_diff * 2)  # Усиливаем важность соответствия
        
        # Оценка по прямоугольности
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        solidity = contour_area / hull_area if hull_area > 0 else 0
        score += solidity * 0.5  # До 0.5 баллов за прямоугольность
        
        # Оценка по углам (должны быть близки к 90 градусам)
        if len(contour) == 4:
            pts = contour.reshape(4, 2).astype(np.float32)
            angles = []
            for i in range(4):
                p1 = pts[i]
                p2 = pts[(i + 1) % 4]
                p3 = pts[(i + 2) % 4]
                
                v1 = p1 - p2
                v2 = p3 - p2
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                angles.append(angle)
            
            # Среднее отклонение от 90 градусов
            angle_deviation = np.mean([abs(a - 90) for a in angles])
            angle_score = max(0, 1.0 - angle_deviation / 45)  # Идеально 90 градусов
            score += angle_score * 0.3
        
        # Бонус за размер (предпочитаем более крупные документы)
        if area_ratio > 0.2:
            score += 0.2
        
        return score
    
    def crop_with_calibration(self, image: np.ndarray) -> np.ndarray:
        """Обрезает изображение используя автоматическое обнаружение с калибровкой"""
        # Автоматически находим документ
        contour = self.find_document_auto(image)
        
        if contour is not None:
            # Выравниваем перспективу
            result = self.four_point_transform(image, contour.reshape(4, 2))
            return result
        else:
            # Fallback: используем сохраненные точки калибровки если они есть
            if (self.calibration_config.crop_points is not None and 
                len(self.calibration_config.crop_points) == 4):
                h, w = image.shape[:2]
                points = [(int(x * w), int(y * h)) for x, y in self.calibration_config.crop_points]
                points_array = np.array(points, dtype=np.float32)
                print("⚠️  Используем сохраненные точки калибровки")
                result = self.four_point_transform(image, points_array)
                return result
            else:
                # Последний fallback: возвращаем оригинал
                print("⚠️  Документ не найден, возвращаем оригинал")
                return image
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Выравнивает перспективу по 4 точкам с улучшенной обработкой"""
        # Упорядочиваем точки
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Вычисляем ширину (берем среднее для большей стабильности)
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        # Вычисляем высоту (берем среднее для большей стабильности)
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Валидация размеров
        if maxWidth < 10 or maxHeight < 10:
            print("⚠️  Слишком маленький размер, возвращаем оригинал")
            return image
        
        # Если есть целевой размер из калибровки, используем его
        if (self.calibration_config.target_size is not None and 
            self.calibration_config.target_size[0] > 0 and 
            self.calibration_config.target_size[1] > 0):
            target_w, target_h = self.calibration_config.target_size
            # Сохраняем пропорции, но используем целевой размер как ориентир
            aspect_ratio = target_w / target_h
            if maxWidth / maxHeight > aspect_ratio:
                maxHeight = int(maxWidth / aspect_ratio)
            else:
                maxWidth = int(maxHeight * aspect_ratio)
        
        # Формируем точки назначения
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        # Вычисляем матрицу преобразования
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Применяем преобразование с улучшенной интерполяцией
        warped = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  # Белый фон
        )
        
        return warped
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """Улучшенное упорядочивание точек: top-left, top-right, bottom-right, bottom-left"""
        # Конвертируем в numpy array если нужно
        pts = np.array(pts, dtype=np.float32)
        
        # Если уже 4x2, используем как есть
        if pts.shape != (4, 2):
            pts = pts.reshape(4, 2)
        
        rect = np.zeros((4, 2), dtype="float32")
        
        # Метод 1: сумма координат (top-left имеет наименьшую сумму, bottom-right - наибольшую)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        
        # Метод 2: разность координат (top-right имеет наименьшую разность, bottom-left - наибольшую)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        
        # Валидация: проверяем что точки действительно образуют прямоугольник
        # Вычисляем углы
        def angle_between_points(p1, p2, p3):
            """Вычисляет угол в точке p2"""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.arccos(cos_angle) * 180 / np.pi
        
        # Проверяем углы (должны быть близки к 90 градусам)
        angles = []
        for i in range(4):
            p1 = rect[i]
            p2 = rect[(i + 1) % 4]
            p3 = rect[(i + 2) % 4]
            angles.append(angle_between_points(p1, p2, p3))
        
        avg_angle = np.mean(angles)
        # Если средний угол сильно отличается от 90, возможно точки перепутаны
        # В этом случае используем альтернативный метод
        
        if abs(avg_angle - 90) > 30:
            # Альтернативный метод: сортировка по углу от центра
            center = pts.mean(axis=0)
            angles_from_center = []
            for pt in pts:
                angle = np.arctan2(pt[1] - center[1], pt[0] - center[0]) * 180 / np.pi
                angles_from_center.append((angle, pt))
            
            # Сортируем по углу
            angles_from_center.sort(key=lambda x: x[0])
            sorted_pts = np.array([pt for _, pt in angles_from_center], dtype=np.float32)
            
            # Находим top-left (наименьшая сумма x+y)
            s = sorted_pts.sum(axis=1)
            top_left_idx = np.argmin(s)
            
            # Переупорядочиваем начиная с top-left
            rect = np.roll(sorted_pts, -top_left_idx, axis=0)
        
        return rect
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """Обрабатывает одно изображение используя калибровку"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Не удалось загрузить: {image_path}")
                return None
            
            original_size = f"{image.shape[1]}x{image.shape[0]}"
            
            # Автоматически находим и обрезаем документ
            result = self.crop_with_calibration(image)
            
            new_size = f"{result.shape[1]}x{result.shape[0]}"
            compression = (result.shape[0] * result.shape[1]) / (image.shape[0] * image.shape[1])
            
            print(f"📄 {Path(image_path).name} {original_size} -> {new_size} ({compression*100:.1f}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка обработки {image_path}: {e}")
            return cv2.imread(image_path)
    
    def process_folder(self, input_folder: str, output_folder: str) -> dict:
        """Обрабатывает папку с изображениями используя калибровку"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']:
            image_files.extend(input_path.glob(ext))
        
        stats = {'total': len(image_files), 'processed': 0, 'failed': 0}
        
        print(f"\n🎯 Обработка {len(image_files)} файлов с автоматическим обнаружением...")
        
        for i, image_file in enumerate(image_files, 1):
            output_file = output_path / f"cropped_{image_file.name}"
            
            if output_file.exists():
                print(f"⏭️ {i:2d}/{len(image_files)}: {image_file.name} (уже обработан)")
                stats['processed'] += 1
                continue
            
            result = self.process_single_image(str(image_file))
            
            if result is not None:
                cv2.imwrite(str(output_file), result, [
                    int(cv2.IMWRITE_JPEG_QUALITY), self.processing_config.jpeg_quality
                ])
                stats['processed'] += 1
                print(f"✅ {i:2d}/{len(image_files)}: {image_file.name}")
            else:
                stats['failed'] += 1
                print(f"❌ {i:2d}/{len(image_files)}: {image_file.name}")
        
        print(f"\n📊 Готово! Успешно: {stats['processed']}/{stats['total']}")
        return stats
