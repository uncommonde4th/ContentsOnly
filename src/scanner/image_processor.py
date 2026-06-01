import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from scanner.neural_detector import NeuralDocumentDetector

class ProcessingConfig:
    """Простая конфигурация обработки"""
    def __init__(self):
        self.jpeg_quality = 95

class CalibratedImageProcessor:
    """Обработчик изображений с использованием параметров калибровки"""
    
    def __init__(self, processing_config: ProcessingConfig, calibration_config):
        self.processing_config = processing_config
        self.calibration_config = calibration_config
        self.neural_detector = NeuralDocumentDetector()
    
    def _validate_points(self, points: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """
        Проверяет валидность предсказанных точек
        Возвращает (валидность, причина невалидности)
        """
        h, w = image_shape[:2]
        
        # Проверка 1: Должно быть 4 точки
        if len(points) != 4:
            return False, f"Ожидалось 4 точки, получено {len(points)}"
        
        # Проверка 2: Точки не должны быть слишком близко
        min_distance = min(h, w) * 0.05  # 5% от минимальной стороны
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_distance:
                    return False, f"Точки {i+1} и {j+1} слишком близко ({dist:.0f}px < {min_distance:.0f}px)"
        
        # Проверка 3: Все точки должны быть в пределах изображения с небольшим запасом
        margin = 50  # небольшой запас для точек на границе
        for i, (x, y) in enumerate(points):
            if x < -margin or x >= w + margin or y < -margin or y >= h + margin:
                return False, f"Точка {i+1} за пределами изображения ({x:.0f}, {y:.0f})"
        
        # Проверка 4: Площадь контура должна быть разумной
        area = cv2.contourArea(points.astype(np.float32))
        image_area = w * h
        area_ratio = area / image_area
        
        if area_ratio < 0.01:  # Меньше 1% изображения
            return False, f"Слишком маленькая площадь ({area_ratio*100:.1f}% изображения)"
        
        if area_ratio > 0.95:  # Больше 95% изображения
            return False, f"Слишком большая площадь ({area_ratio*100:.1f}% изображения)"
        
        # Все проверки пройдены
        return True, "OK"

    def _fix_duplicate_points(self, points: np.ndarray) -> np.ndarray:
        """
        Исправляет дублирующиеся точки (когда модель предсказывает 3 уникальные точки)
        """
        # Проверяем на дубликаты
        unique_points = []
        for point in points:
            is_duplicate = False
            for unique_point in unique_points:
                if np.linalg.norm(point - unique_point) < 10:  # расстояние меньше 10 пикселей
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        # Если есть дубликаты, пробуем их исправить
        if len(unique_points) == 3:
            print(f"      ⚠️ Обнаружено дублирование точек, исправляем...")
            
            # Сортируем уникальные точки по углу от центра
            center = np.mean(unique_points, axis=0)
            angles = np.arctan2(
                [p[1] - center[1] for p in unique_points],
                [p[0] - center[0] for p in unique_points]
            )
            sorted_idx = np.argsort(angles)
            sorted_points = [unique_points[i] for i in sorted_idx]
            
            # Создаём 4 точки, вставляя среднюю точку между соседними
            fixed_points = []
            for i in range(3):
                fixed_points.append(sorted_points[i])
                # Добавляем среднюю точку между i и i+1
                next_point = sorted_points[(i + 1) % 3]
                mid_point = (sorted_points[i] + next_point) / 2
                fixed_points.append(mid_point)
            
            return np.array(fixed_points[:4])
        
        return points

    def find_document_auto(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Автоматически находит документ на изображении
        Сначала пробует нейросеть, затем эвристические методы
        """
        
        # ========== ШАГ 1: НЕЙРОСЕТЬ ==========
        if self.neural_detector.is_available:
            print("🧠 Пробуем нейросеть...")
            
            # Логируем информацию об изображении
            h, w = image.shape[:2]
            print(f"   Размер изображения: {w}x{h}")
            print(f"   Соотношение сторон: {w/h:.2f}")
            
            # Пробуем нейросеть с разными порогами уверенности
            for conf in [0.7, 0.5, 0.3, 0.2, 0.1]:
                print(f"   Пробуем порог уверенности: {conf}")
                
                points = self.neural_detector.detect_document(image, conf_threshold=conf)
                
                if points is None:
                    print(f"      ❌ Нет предсказаний (уверенность ниже {conf})")
                    continue
                
                print(f"      ✅ Получены точки: {points}")
                
                # Исправляем дубликаты
                points = self._fix_duplicate_points(points)
                
                # Проверяем валидность точек
                is_valid, reason = self._validate_points(points, image.shape)
                
                if is_valid:
                    contour = points.reshape(-1, 1, 2)
                    print(f"   ✅ Документ найден нейросетью (уверенность: {conf})")
                    print(f"   📍 Точки: {points.tolist()}")
                    return contour
                else:
                    print(f"      ⚠️ Точки невалидны: {reason}")
                    print(f"      📍 Полученные точки: {points.tolist()}")
                    continue
            
            print("   ❌ Нейросеть не смогла найти валидный документ ни с одним порогом")
        else:
            print("⚠️ Нейросеть недоступна (модель не загружена)")
            print("   Проверьте наличие файла models/doc_detector.pt")
        
        # ========== ШАГ 2: ЭВРИСТИЧЕСКИЕ МЕТОДЫ (FALLBACK) ==========
        print("⚠️ Переключаемся на эвристические методы...")
        
        # Метод 0: Специальный метод для светлых документов на темном фоне
        contour = self._find_light_on_dark(image)
        if contour is not None:
            print("✅ Найден как светлый документ на темном фоне")
            return contour
        
        # Метод 0.5: Поиск краев документа
        contour = self._find_document_edges(image)
        if contour is not None:
            print("✅ Найден по краям документа")
            return contour
        
        # Метод 1: Поиск по краям
        contour = self._find_by_edges(image)
        if contour is not None:
            print("✅ Найден по краям")
            return contour
        
        # Метод 2: Поиск по цвету
        contour = self._find_by_color(image)
        if contour is not None:
            print("✅ Найден по цвету")
            return contour
        
        # Метод 3: Поиск по текстурам
        contour = self._find_by_texture(image)
        if contour is not None:
            print("✅ Найден по текстуре")
            return contour
        
        # Метод 4: Ослабленные ограничения
        print("⚠️ Попытка с ослабленными ограничениями...")
        contour = self._find_with_relaxed_constraints(image)
        if contour is not None:
            print("✅ Найден с ослабленными ограничениями")
            return contour
        
        # Метод 5: Любой большой прямоугольник
        print("⚠️ Поиск любого большого прямоугольного контура...")
        contour = self._find_any_large_rectangle(image)
        if contour is not None:
            print("✅ Найден большой прямоугольный контур")
            return contour
        
        print("❌ Документ не найден автоматически")
        return None
    
    def _find_light_on_dark(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Специальный метод для поиска светлых документов на темном фоне, используя ВСЕ данные калибровки"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Пробуем несколько методов бинаризации
        binaries = []
        
        # Метод 1: Используем информацию из калибровки
        if self.calibration_config.avg_color is not None and self.calibration_config.avg_bg_color is not None:
            doc_brightness = np.mean(self.calibration_config.avg_color)
            bg_brightness = np.mean(self.calibration_config.avg_bg_color)
            
            # Используем образцы фона из калибровки
            if len(self.calibration_config.bg_samples) > 0:
                bg_brightnesses = [np.mean(sample) for sample in self.calibration_config.bg_samples]
                bg_brightness = np.mean(bg_brightnesses)
            
            if doc_brightness > bg_brightness + 20:
                # Пробуем несколько порогов вокруг среднего (более широкий диапазон)
                for offset in [-15, -10, -5, 0, 5, 10, 15]:
                    threshold_value = int((doc_brightness + bg_brightness) / 2) + offset
                    threshold_value = max(60, min(240, threshold_value))
                    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                    binaries.append(binary)
        
        # Метод 2: Otsu автоматический порог
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binaries.append(binary_otsu)
        
        # Метод 3: Адаптивный порог (несколько вариантов)
        for block_size in [11, 15, 21]:
            binary_adaptive = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 
                block_size, 10
            )
            binaries.append(binary_adaptive)
        
        # Пробуем каждый метод бинаризации
        for binary in binaries:
            # Морфологические операции для улучшения маски
            # ВАЖНО: Более агрессивная обработка чтобы захватить весь документ, а не только текст
            kernel_small = np.ones((3, 3), np.uint8)
            kernel_medium = np.ones((7, 7), np.uint8)
            kernel_large = np.ones((11, 11), np.uint8)
            
            # Удаляем мелкий шум
            binary_clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
            # Заполняем пробелы внутри документа (более агрессивно)
            binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_CLOSE, kernel_medium, iterations=4)
            # Расширяем чтобы захватить края документа и поля
            binary_clean = cv2.dilate(binary_clean, kernel_large, iterations=3)
            
            # Сначала пробуем строгий режим
            contour = self._find_best_contour(binary_clean, image.shape, strict=True)
            if contour is not None:
                # Расширяем контур чтобы захватить весь документ с полями
                contour = self._expand_contour_slightly(contour, image.shape)
                # Проверяем что контур разумный
                if self._validate_contour(contour, image.shape):
                    return contour
            
            # Если строгий режим не нашел, пробуем нестрогий режим с поддержкой вертикальных документов
            contour = self._find_best_contour(binary_clean, image.shape, strict=False, allow_vertical=True)
            if contour is not None:
                # Расширяем контур чтобы захватить весь документ с полями
                contour = self._expand_contour_slightly(contour, image.shape)
                # Проверяем что контур разумный
                if self._validate_contour(contour, image.shape):
                    return contour
        
        return None
    
    def _find_document_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Специальный метод для поиска краев документа (не текста, а краев бумаги)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Используем информацию из калибровки о цвете документа
        if self.calibration_config.avg_color is not None and self.calibration_config.avg_bg_color is not None:
            doc_brightness = np.mean(self.calibration_config.avg_color)
            bg_brightness = np.mean(self.calibration_config.avg_bg_color)
            
            if len(self.calibration_config.bg_samples) > 0:
                bg_brightnesses = [np.mean(sample) for sample in self.calibration_config.bg_samples]
                bg_brightness = np.mean(bg_brightnesses)
            
            # Если документ светлее фона, используем бинаризацию
            if doc_brightness > bg_brightness + 15:
                # Используем более низкий порог чтобы захватить весь документ, включая поля
                threshold_value = int(bg_brightness + (doc_brightness - bg_brightness) * 0.3)
                threshold_value = max(70, min(220, threshold_value))
                
                _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            else:
                # Fallback на Otsu
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Fallback на Otsu
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # ОЧЕНЬ агрессивная морфологическая обработка чтобы захватить весь документ
        kernel_medium = np.ones((15, 15), np.uint8)
        kernel_large = np.ones((25, 25), np.uint8)
        
        # Заполняем все пробелы внутри документа (включая пробелы между строками текста)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=5)
        # Расширяем чтобы захватить края документа
        binary = cv2.dilate(binary, kernel_medium, iterations=4)
        
        # Ищем контуры
        contour = self._find_best_contour(binary, image.shape, strict=True)
        if contour is not None:
            # Расширяем контур чтобы захватить весь документ с полями
            contour = self._expand_contour_slightly(contour, image.shape)
            if self._validate_contour(contour, image.shape):
                return contour
        
        return None
    
    def _expand_contour_slightly(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Расширяет контур чтобы захватить весь документ с полями и фоном (увеличивает размер на 15-20%)"""
        h, w = image_shape[:2]
        
        try:
            # Преобразуем контур в массив точек
            if len(contour.shape) == 3:
                pts = contour.reshape(-1, 2).astype(np.float32)
            else:
                pts = contour.astype(np.float32)
            
            if len(pts) < 4:
                return contour
            
            # Вычисляем размеры контура
            x_coords = pts[:, 0]
            y_coords = pts[:, 1]
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            
            # Определяем соотношение сторон для длинных документов
            aspect_ratio = height / width if width > 0 else 1.0
            is_vertical = aspect_ratio > 2.0
            
            # УВЕЛИЧИВАЕМ расширение: 15-20% от размера контура
            # Для вертикальных документов используем больший отступ
            expand_ratio_x = 0.18 if is_vertical else 0.15
            expand_ratio_y = 0.20 if is_vertical else 0.15
            
            # Минимум 25 пикселей, максимум 80 для длинных документов
            expand_x = max(25, min(80 if is_vertical else 60, int(width * expand_ratio_x)))
            expand_y = max(25, min(100 if is_vertical else 80, int(height * expand_ratio_y)))
            
            # Вычисляем центр
            center_x = (np.min(x_coords) + np.max(x_coords)) / 2
            center_y = (np.min(y_coords) + np.max(y_coords)) / 2
            
            # Расширяем каждую точку от центра пропорционально
            expanded_pts = pts.copy()
            for i in range(len(expanded_pts)):
                dx = expanded_pts[i, 0] - center_x
                dy = expanded_pts[i, 1] - center_y
                
                # Расширяем пропорционально
                if abs(dx) > 0.1:
                    expanded_pts[i, 0] += (dx / abs(dx)) * expand_x if dx != 0 else 0
                if abs(dy) > 0.1:
                    expanded_pts[i, 1] += (dy / abs(dy)) * expand_y if dy != 0 else 0
            
            # Ограничиваем точками в пределах изображения
            expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w - 1)
            expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h - 1)
            
            # Возвращаем в исходном формате
            if len(contour.shape) == 3:
                return expanded_pts.reshape(-1, 1, 2).astype(np.int32)
            else:
                return expanded_pts.astype(np.int32)
        except:
            # Если ошибка, возвращаем оригинал
            return contour
    
    def _validate_contour(self, contour: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Проверяет что контур валиден и разумен"""
        h, w = image_shape[:2]
        
        if contour is None or len(contour) < 4:
            return False
        
        try:
            contour_reshaped = contour.reshape(4, 2)
            
            # Проверяем что точки не слишком близки
            min_distance = min(h, w) * 0.05
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = np.linalg.norm(contour_reshaped[i] - contour_reshaped[j])
                    if dist < min_distance:
                        return False
            
            # Проверяем что контур находится в пределах изображения
            for point in contour_reshaped:
                if point[0] < 0 or point[0] >= w or point[1] < 0 or point[1] >= h:
                    return False
            
            # Проверяем площадь контура
            area = cv2.contourArea(contour_reshaped)
            image_area = w * h
            if area < image_area * 0.01:  # Минимум 1% изображения
                return False
            
            return True
        except:
            return False
    
    def _find_by_color(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Поиск документа по цвету используя ВСЮ расширенную информацию калибровки"""
        if self.calibration_config.avg_color is None:
            return None
        
        # Конвертируем в LAB цветовое пространство (более восприимчиво к изменениям освещения)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_color_lab = cv2.cvtColor(
            np.uint8([[self.calibration_config.avg_color]]), 
            cv2.COLOR_BGR2LAB
        )[0][0]
        
        # Используем расширенную информацию о цвете документа если доступна
        if self.calibration_config.document_color_std is not None:
            # Используем стандартное отклонение для более точного определения
            color_std_lab = cv2.cvtColor(
                np.uint8([[self.calibration_config.document_color_std]]), 
                cv2.COLOR_BGR2LAB
            )[0][0]
            # Используем 2.5 стандартных отклонения для диапазона
            threshold_range = np.linalg.norm(color_std_lab) * 2.5
        else:
            # Fallback на базовый порог из калибровки
            threshold_range = self.calibration_config.color_threshold * 1.5
        
        # Вычисляем разницу в LAB пространстве (более точная метрика)
        color_diff = np.linalg.norm(
            image_lab.astype(np.float32) - avg_color_lab.astype(np.float32), 
            axis=2
        )
        
        # ДОПОЛНИТЕЛЬНО: Используем информацию о фоне для улучшения детекции
        # Исключаем области которые похожи на фон
        if self.calibration_config.avg_bg_color is not None and len(self.calibration_config.bg_samples) > 0:
            # Вычисляем разницу с фоном
            avg_bg_color_lab = cv2.cvtColor(
                np.uint8([[self.calibration_config.avg_bg_color]]), 
                cv2.COLOR_BGR2LAB
            )[0][0]
            
            bg_diff = np.linalg.norm(
                image_lab.astype(np.float32) - avg_bg_color_lab.astype(np.float32), 
                axis=2
            )
            
            # Если пиксель ближе к фону чем к документу, исключаем его
            bg_threshold = self.calibration_config.color_threshold * 0.8
            bg_mask = bg_diff < bg_threshold
            color_diff[bg_mask] = 255  # Помечаем как фон (будет исключено)
        
        # Адаптивный порог с учетом расширенной информации
        threshold = max(20, min(80, threshold_range))
        
        # Бинаризация
        _, binary = cv2.threshold(
            color_diff.astype(np.uint8), 
            int(threshold), 
            255, 
            cv2.THRESH_BINARY_INV
        )
        
        # Улучшенные морфологические операции (более агрессивные чтобы захватить весь документ)
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((7, 7), np.uint8)
        kernel_large = np.ones((11, 11), np.uint8)
        
        # Удаляем шум
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        # Заполняем пробелы (более агрессивно)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
        # Расширяем чтобы захватить поля документа
        binary = cv2.dilate(binary, kernel_medium, iterations=2)
        
        contour = self._find_best_contour(binary, image.shape, strict=True)
        if contour is not None:
            contour = self._expand_contour_slightly(contour, image.shape)
            if self._validate_contour(contour, image.shape):
                return contour
        return None
    
    def _find_by_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Улучшенный поиск документа по краям"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Предобработка: уменьшаем шум
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Пробуем несколько наборов параметров Canny для большей надежности
        edge_results = []
        
        # Набор 1: Адаптивные параметры на основе калибровки
        if self.calibration_config.edge_threshold > 0:
            low_threshold = max(30, int(self.calibration_config.edge_threshold * 0.5))
            high_threshold = min(200, int(self.calibration_config.edge_threshold * 1.5))
            edges1 = cv2.Canny(gray, low_threshold, high_threshold)
            edge_results.append(edges1)
        
        # Набор 2: Автоматический выбор порогов
        median = np.median(gray)
        low_threshold = int(max(0, 0.7 * median))
        high_threshold = int(min(255, 1.3 * median))
        edges2 = cv2.Canny(gray, low_threshold, high_threshold)
        edge_results.append(edges2)
        
        # Набор 3: Более чувствительные пороги для слабых краев
        low_threshold = int(max(0, 0.5 * median))
        high_threshold = int(min(255, 1.5 * median))
        edges3 = cv2.Canny(gray, low_threshold, high_threshold)
        edge_results.append(edges3)
        
        # Набор 4: Фиксированные пороги для документов на темном фоне
        edges4 = cv2.Canny(gray, 50, 150)
        edge_results.append(edges4)
        
        # Пробуем каждый набор краев
        for edges in edge_results:
            # Улучшаем края: соединяем близкие линии
            kernel = np.ones((5, 5), np.uint8)
            edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            edges_processed = cv2.dilate(edges_processed, kernel, iterations=1)
            
            contour = self._find_best_contour(edges_processed, image.shape, strict=True)
            if contour is not None:
                # Расширяем контур чтобы захватить весь документ
                contour = self._expand_contour_slightly(contour, image.shape)
                if self._validate_contour(contour, image.shape):
                    return contour
        
        # Если ничего не нашли, пробуем с ослабленными ограничениями
        for edges in edge_results:
            kernel = np.ones((7, 7), np.uint8)
            edges_processed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
            edges_processed = cv2.dilate(edges_processed, kernel, iterations=2)
            
            contour = self._find_best_contour(edges_processed, image.shape, strict=False)
            if contour is not None:
                # Расширяем контур чтобы захватить весь документ
                contour = self._expand_contour_slightly(contour, image.shape)
                if self._validate_contour(contour, image.shape):
                    return contour
        
        return None
    
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
        
        contour = self._find_best_contour(binary, image.shape, strict=True)
        if contour is not None:
            contour = self._expand_contour_slightly(contour, image.shape)
            if self._validate_contour(contour, image.shape):
                return contour
        return None
    
    def _find_best_contour(self, binary: np.ndarray, image_shape: Tuple[int, int], 
                          strict: bool = True, allow_vertical: bool = False) -> Optional[np.ndarray]:
        """Находит лучший контур удовлетворяющий параметрам калибровки
        
        Args:
            binary: Бинарное изображение
            image_shape: Размеры изображения (h, w)
            strict: Использовать строгие проверки калибровки
            allow_vertical: Разрешить вертикально вытянутые документы (до 10:1)
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Сортируем контуры по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        h, w = image_shape[:2]
        image_area = w * h
        best_contour = None
        best_score = -1  # Начинаем с -1 чтобы принимать контуры с score = 0
        
        # Проверяем до 20 крупнейших контуров
        for idx, contour in enumerate(contours[:20]):
            area = cv2.contourArea(contour)
            
            # Минимальная площадь (хотя бы 2% изображения - очень мягкое требование)
            if area < image_area * 0.02:
                continue
            
            area_ratio = area / image_area
            
            # Пробуем разные epsilon для аппроксимации
            found_valid_approx = False
            approx = None
            for eps_factor in [0.01, 0.015, 0.02, 0.03, 0.05]:
                epsilon = eps_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Нужно минимум 4 точки для прямоугольника
                if len(approx) < 4:
                    continue
                
                # Если больше 4 точек, пытаемся упростить
                if len(approx) > 4:
                    # Пробуем более агрессивное упрощение
                    for eps_factor2 in [0.05, 0.08, 0.1]:
                        epsilon2 = eps_factor2 * cv2.arcLength(contour, True)
                        approx2 = cv2.approxPolyDP(contour, epsilon2, True)
                        if len(approx2) == 4:
                            approx = approx2
                            break
                    
                    # Если все еще больше 4, берем первые 4 или используем выпуклую оболочку
                    if len(approx) > 4:
                        hull = cv2.convexHull(contour)
                        if len(hull) >= 4:
                            # Берем 4 точки из выпуклой оболочки
                            if len(hull) == 4:
                                approx = hull
                            else:
                                # Выбираем 4 точки с максимальными расстояниями
                                approx = hull[:4]  # Просто берем первые 4
                
                # Проверяем выпуклость (более мягкая проверка)
                try:
                    is_convex = cv2.isContourConvex(approx)
                    if not is_convex:
                        # Проверяем solidity - если достаточно выпуклый, принимаем
                        hull = cv2.convexHull(approx)
                        hull_area = cv2.contourArea(hull)
                        contour_area = cv2.contourArea(approx)
                        if hull_area > 0:
                            solidity = contour_area / hull_area
                            if solidity < 0.85:  # Недостаточно выпуклый
                                continue
                except:
                    # Если ошибка при проверке, пропускаем этот epsilon
                    continue
                
                # Если дошли сюда, у нас есть валидный approx
                found_valid_approx = True
                break  # Выходим из цикла epsilon
            
            # Если не нашли валидный approx, пропускаем этот контур
            if not found_valid_approx or approx is None:
                continue
            
            # Вычисляем соотношение сторон для проверки
            rect = cv2.minAreaRect(approx)
            width, height = rect[1]
            if min(width, height) < 10:  # Слишком маленький
                continue
                
            aspect_ratio = max(width, height) / min(width, height)
            
            if strict:
                # Строгие проверки с калибровкой
                # Проверяем площадь (с более мягкими границами если диапазон слишком узкий)
                area_min, area_max = self.calibration_config.area_range
                if area_min > 0 and area_max > 0:
                    # Расширяем диапазон на 20% для большей гибкости
                    area_min = max(0.05, area_min * 0.8)
                    area_max = min(0.95, area_max * 1.2)
                    if not (area_min <= area_ratio <= area_max):
                        continue
                
                # Проверяем соотношение сторон (с поддержкой вертикальных документов)
                aspect_min, aspect_max = self.calibration_config.aspect_ratio_range
                if aspect_min > 0 and aspect_max > 0:
                    # Если разрешены вертикальные документы, значительно расширяем диапазон
                    if allow_vertical:
                        # Для вертикальных документов: расширяем до 10:1
                        aspect_min = max(1.0, aspect_min * 0.5)  # Еще более мягкая нижняя граница
                        aspect_max = min(10.0, max(aspect_max * 2.0, 8.0))  # Разрешаем до 10:1
                    else:
                        # Обычная логика с более мягкими границами
                        aspect_min = max(1.0, aspect_min * 0.7)
                        aspect_max = min(10.0, aspect_max * 1.3)
                    
                    if not (aspect_min <= aspect_ratio <= aspect_max):
                        continue
            else:
                # Ослабленные проверки - только базовые требования
                if min(width, height) < 20:  # Минимальный размер
                    continue
                
                # Для вертикальных документов разрешаем до 10:1, иначе до 5:1
                max_aspect = 10.0 if allow_vertical else 5.0
                if aspect_ratio > max_aspect:
                    continue
            
            # Оцениваем контур
            if strict:
                rect = cv2.minAreaRect(approx)
                width, height = rect[1]
                aspect_ratio = max(width, height) / min(width, height)
                score = self._score_contour(approx, area_ratio, aspect_ratio, image_area)
            else:
                # Простая оценка для ослабленного режима
                score = area_ratio  # Просто предпочитаем большие контуры
            
            if score > best_score:
                best_score = score
                best_contour = approx
        
        return best_contour
    
    def _find_with_relaxed_constraints(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Поиск с ослабленными ограничениями калибровки"""
        # Пробуем все методы с ослабленными ограничениями
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Адаптивный Canny
        median = np.median(gray)
        low_threshold = int(max(0, 0.5 * median))
        high_threshold = int(min(255, 1.5 * median))
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Более агрессивная обработка краев
        kernel = np.ones((7, 7), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contour = self._find_best_contour(edges, image.shape, strict=False)
        if contour is not None:
            # Расширяем контур чтобы захватить весь документ
            contour = self._expand_contour_slightly(contour, image.shape)
            if self._validate_contour(contour, image.shape):
                return contour
        return None
    
    def _find_any_large_rectangle(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Находит любой большой прямоугольный контур без ограничений калибровки"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Предобработка
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        h, w = image.shape[:2]
        image_area = w * h
        
        # Множественные попытки с разными параметрами Canny
        for low, high in [(30, 100), (50, 150), (70, 200), (100, 250)]:
            edges = cv2.Canny(gray, low, high)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Проверяем больше контуров (до 10)
            for contour in contours[:10]:
                area = cv2.contourArea(contour)
                # Более мягкое требование к площади - минимум 5% изображения
                if area < image_area * 0.05:
                    continue
                
                # Пробуем разные epsilon для аппроксимации (более широкий диапазон)
                for eps_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]:
                    epsilon = eps_factor * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Принимаем контуры с 4 точками (идеально) или близкие к 4
                    if len(approx) >= 4:
                        # Если больше 4 точек, пытаемся упростить еще больше
                        if len(approx) > 4:
                            epsilon = 0.1 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            if len(approx) != 4:
                                # Берем первые 4 точки если не удалось упростить
                                if len(approx) > 4:
                                    # Находим 4 угловые точки
                                    hull = cv2.convexHull(contour)
                                    if len(hull) >= 4:
                                        # Берем 4 точки из выпуклой оболочки
                                        # Выбираем точки с максимальными расстояниями
                                        distances = []
                                        for i in range(len(hull)):
                                            for j in range(i+1, len(hull)):
                                                dist = np.linalg.norm(hull[i] - hull[j])
                                                distances.append((dist, i, j))
                                        distances.sort(reverse=True)
                                        # Берем 4 точки которые образуют наибольшие расстояния
                                        selected_indices = set()
                                        for dist, i, j in distances[:6]:
                                            selected_indices.add(i)
                                            selected_indices.add(j)
                                            if len(selected_indices) >= 4:
                                                break
                                        if len(selected_indices) >= 4:
                                            approx = hull[list(selected_indices)[:4]]
                        
                        # Проверяем что это разумный контур
                        if len(approx) >= 4:
                            # Проверяем выпуклость (если возможно)
                            try:
                                if not cv2.isContourConvex(approx):
                                    # Если не выпуклый, проверяем что это не слишком плохо
                                    hull = cv2.convexHull(approx)
                                    hull_area = cv2.contourArea(hull)
                                    contour_area = cv2.contourArea(approx)
                                    if hull_area > 0:
                                        solidity = contour_area / hull_area
                                        if solidity < 0.7:  # Слишком невыпуклый
                                            continue
                            except:
                                pass
                            
                            # Берем первые 4 точки
                            if len(approx) > 4:
                                approx = approx[:4]
                            
                            # Проверяем размеры
                            rect = cv2.minAreaRect(approx)
                            width, height = rect[1]
                            if min(width, height) < 20:  # Минимальный размер
                                continue
                            
                            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
                            if aspect_ratio > 10.0:  # Не слишком вытянутый
                                continue
                            
                            # Если дошли сюда - нашли подходящий контур
                            contour_result = approx.reshape(-1, 1, 2) if len(approx.shape) == 2 else approx
                            # Расширяем контур чтобы захватить весь документ
                            contour_result = self._expand_contour_slightly(contour_result, image.shape)
                            if self._validate_contour(contour_result, image.shape):
                                return contour_result
        
        # Если не нашли через края, пробуем через адаптивный порог
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 15, 5)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours[:5]:
                area = cv2.contourArea(contour)
                if area < image_area * 0.05:
                    continue
                
                epsilon = 0.05 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:
                    if len(approx) > 4:
                        approx = approx[:4]
                    rect = cv2.minAreaRect(approx)
                    width, height = rect[1]
                    if min(width, height) >= 20:
                        contour_result = approx.reshape(-1, 1, 2) if len(approx.shape) == 2 else approx
                        # Расширяем контур чтобы захватить весь документ
                        contour_result = self._expand_contour_slightly(contour_result, image.shape)
                        if self._validate_contour(contour_result, image.shape):
                            return contour_result
        
        return None
    
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
        h, w = image.shape[:2]
        
        # Автоматически находим документ
        contour = self.find_document_auto(image)
        
        if contour is not None:
            # Проверяем что контур разумный перед использованием
            contour_reshaped = contour.reshape(4, 2)
            
            # Проверяем что точки не слишком близки друг к другу
            min_distance = min(h, w) * 0.05  # Минимум 5% от размера изображения
            valid = True
            for i in range(4):
                for j in range(i + 1, 4):
                    dist = np.linalg.norm(contour_reshaped[i] - contour_reshaped[j])
                    if dist < min_distance:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                print("⚠️  Найденный контур невалиден (точки слишком близки)")
                contour = None  # Продолжаем к fallback
            
            if contour is not None:
                # Просто обрезаем по прямоугольнику (без выравнивания перспективы)
                result = self.rectangular_crop(image, contour_reshaped)
                return result
        else:
            # Fallback: используем сохраненные точки калибровки если они есть
            if (self.calibration_config.crop_points is not None and 
                len(self.calibration_config.crop_points) == 4):
                h, w = image.shape[:2]
                points = [(int(x * w), int(y * h)) for x, y in self.calibration_config.crop_points]
                points_array = np.array(points, dtype=np.float32)
                
                # Проверяем что точки находятся в пределах изображения
                points_array[:, 0] = np.clip(points_array[:, 0], 0, w - 1)
                points_array[:, 1] = np.clip(points_array[:, 1], 0, h - 1)
                
                print("⚠️  Используем сохраненные точки калибровки (адаптированные к размеру изображения)")
                result = self.rectangular_crop(image, points_array)
                return result
            else:
                # Последний fallback: пытаемся найти любой документ без калибровки
                print("⚠️  Попытка найти документ без калибровки...")
                contour = self._find_any_large_rectangle(image)
                if contour is not None:
                    print("✅ Найден документ без калибровки")
                    result = self.rectangular_crop(image, contour.reshape(4, 2))
                    return result
                else:
                    print("⚠️  Документ не найден, возвращаем оригинал")
                    return image
    
    def rectangular_crop(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Обрезает изображение по прямоугольнику с отступами (без выравнивания перспективы)"""
        h, w = image.shape[:2]
        
        # Находим ограничивающий прямоугольник
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        
        x_min = int(np.min(x_coords))
        x_max = int(np.max(x_coords))
        y_min = int(np.min(y_coords))
        y_max = int(np.max(y_coords))
        
        # Вычисляем размеры
        width = x_max - x_min
        height = y_max - y_min
        
        # Вычисляем отступ (2% от размера или минимум 15 пикселей)
        margin_ratio = 0.02 if height / width > 2.0 else 0.015  # Больше отступ для вертикальных документов
        margin_x = max(15, int(width * margin_ratio))
        margin_y = max(15, int(height * margin_ratio))
        
        # Применяем отступы
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        
        # Обрезаем изображение
        cropped = image[y_min:y_max, x_min:x_max]
        
        print(f"📐 Обрезка: {width}x{height} -> {x_max-x_min}x{y_max-y_min} (отступ {margin_x}x{margin_y})")
        
        return cropped
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Выравнивает перспективу по 4 точкам, вычисляя размеры из найденных точек с отступами"""
        # Упорядочиваем точки
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # ВЫЧИСЛЯЕМ размеры из найденных точек (не используем калибровку для размеров!)
        # Для трапеции берем МАКСИМАЛЬНЫЕ размеры чтобы ничего не обрезать
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        # Валидация: проверяем что размеры разумные (используем калибровку только для валидации)
        if (self.calibration_config.target_size is not None and 
            self.calibration_config.target_size[0] > 0 and 
            self.calibration_config.target_size[1] > 0):
            target_w, target_h = self.calibration_config.target_size
            
            # Проверяем что вычисленные размеры не слишком отличаются от калибровочных
            # (допускаем отклонение до 50% в любую сторону)
            aspect_ratio_calc = maxWidth / maxHeight if maxHeight > 0 else 1.0
            aspect_ratio_target = target_w / target_h if target_h > 0 else 1.0
            
            # Если соотношение сторон сильно отличается, возможно ошибка детекции
            if abs(aspect_ratio_calc - aspect_ratio_target) / aspect_ratio_target > 0.5:
                print(f"⚠️  Предупреждение: соотношение сторон сильно отличается от калибровки ({aspect_ratio_calc:.2f} vs {aspect_ratio_target:.2f})")
        
        # Валидация размеров
        if maxWidth < 10 or maxHeight < 10:
            print("⚠️  Слишком маленький размер, возвращаем оригинал")
            return image
        
        # Вычисляем отступ (2% от размера или минимум 15 пикселей для гарантии видимости фона)
        # Для длинных документов используем больший отступ
        margin_ratio = 0.02 if maxHeight / maxWidth > 2.0 else 0.015  # Больше отступ для вертикальных документов
        margin_x = max(15, int(maxWidth * margin_ratio))
        margin_y = max(15, int(maxHeight * margin_ratio))
        
        # Смещаем исходные точки наружу, чтобы захватить фон вокруг документа
        # Вычисляем центр документа
        center_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4.0
        center_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4.0
        
        # Вычисляем средние размеры для определения смещения
        avg_width = (widthA + widthB) / 2.0
        avg_height = (heightA + heightB) / 2.0
        
        # Смещаем каждую точку наружу от центра пропорционально
        expand_factor_x = margin_x / (avg_width / 2.0) if avg_width > 0 else 0.02
        expand_factor_y = margin_y / (avg_height / 2.0) if avg_height > 0 else 0.02
        
        # Расширяем точки наружу
        expanded_rect = rect.copy()
        for i in range(4):
            dx = rect[i][0] - center_x
            dy = rect[i][1] - center_y
            expanded_rect[i][0] = rect[i][0] + dx * expand_factor_x
            expanded_rect[i][1] = rect[i][1] + dy * expand_factor_y
        
        # Увеличиваем размеры вывода на отступы
        output_width = maxWidth + 2 * margin_x
        output_height = maxHeight + 2 * margin_y
        
        print(f"📐 Вычислены размеры: {maxWidth}x{maxHeight}, выход: {output_width}x{output_height} (отступ {margin_x}x{margin_y})")
        
        # Формируем точки назначения для прямоугольника с отступом
        dst = np.array([
            [margin_x, margin_y],
            [output_width - 1 - margin_x, margin_y],
            [output_width - 1 - margin_x, output_height - 1 - margin_y],
            [margin_x, output_height - 1 - margin_y]], dtype="float32")
        
        # Вычисляем матрицу преобразования из расширенных точек
        M = cv2.getPerspectiveTransform(expanded_rect, dst)
        
        # Применяем преобразование с улучшенной интерполяцией
        # Используем BORDER_CONSTANT с белым фоном для областей вне документа
        warped = cv2.warpPerspective(
            image, M, (output_width, output_height),
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
            
            return self.process_single_image_from_array(image, image_path)
            
        except Exception as e:
            print(f"❌ Ошибка обработки {image_path}: {e}")
            return cv2.imread(image_path)
    
    def process_single_image_from_array(self, image: np.ndarray, image_path: str = "") -> Optional[np.ndarray]:
        """Обрабатывает уже загруженное изображение используя калибровку"""
        try:
            original_size = f"{image.shape[1]}x{image.shape[0]}"
            
            # Автоматически находим и обрезаем документ
            result = self.crop_with_calibration(image)
            
            if result is None:
                return None
            
            new_size = f"{result.shape[1]}x{result.shape[0]}"
            compression = (result.shape[0] * result.shape[1]) / (image.shape[0] * image.shape[1])
            
            filename = Path(image_path).name if image_path else "изображение"
            print(f"📄 {filename} {original_size} -> {new_size} ({compression*100:.1f}%)")
            
            return result
            
        except Exception as e:
            filename = Path(image_path).name if image_path else "изображение"
            print(f"❌ Ошибка обработки {filename}: {e}")
            return image  # Возвращаем оригинал при ошибке
    
    def process_folder(self, input_folder: str, output_folder: str, 
                      calibration_manager=None, progress_callback=None, overwrite=True) -> dict:
        """Обрабатывает папку с изображениями используя калибровку
        
        Args:
            input_folder: Папка с входными изображениями
            output_folder: Папка для сохранения результатов
            calibration_manager: Менеджер калибровки для выбора подходящей ячейки
            progress_callback: Функция для обновления прогресса (current, total, filename)
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG', '*.bmp', '*.BMP', '*.tiff', '*.TIFF']:
            image_files.extend(input_path.glob(ext))
        
        stats = {'total': len(image_files), 'processed': 0, 'failed': 0, 'skipped': 0}
        
        print(f"\n🎯 Обработка {len(image_files)} файлов с автоматическим обнаружением...")
        
        for i, image_file in enumerate(image_files, 1):
            # Обновляем прогресс
            if progress_callback:
                progress_callback(i, len(image_files), image_file.name)
            
            # Нормализуем имя файла как в manual_crop
            original_filename = image_file.name
            print(f"Оригинальное имя файла: {original_filename}")
            
            # Нормализуем имя файла
            final_filename = original_filename
            print(f"Финальное имя: {final_filename}")
            
            output_file = output_path / final_filename
            
            # Проверяем существование файла если перезапись отключена
            if not overwrite and output_file.exists():
                stats['skipped'] += 1
                print(f"⏭️  {i:2d}/{len(image_files)}: {final_filename} (пропущен, файл существует)")
                continue
            
            # Загружаем изображение один раз
            image = cv2.imread(str(image_file))
            if image is None:
                stats['failed'] += 1
                print(f"❌ {i:2d}/{len(image_files)}: {final_filename} (не удалось загрузить)")
                continue
            
            # Если есть менеджер калибровки, выбираем подходящую ячейку для каждого изображения
            old_config = None
            if calibration_manager:
                # Получаем подходящую калибровку
                best_config = calibration_manager.get_best_calibration_for_image(image)
                if best_config:
                    # Временно заменяем калибровку
                    old_config = self.calibration_config
                    self.calibration_config = best_config
            
            # Убираем проверку существования - файлы будут перезаписываться
            was_existing = output_file.exists()
            
            # Обрабатываем изображение (передаем уже загруженное)
            result = self.process_single_image_from_array(image, str(image_file))
            
            # Восстанавливаем старую калибровку
            if old_config is not None:
                self.calibration_config = old_config
            
            if result is not None:
                cv2.imwrite(str(output_file), result, [
                    int(cv2.IMWRITE_JPEG_QUALITY), self.processing_config.jpeg_quality
                ])
                stats['processed'] += 1
                if was_existing:
                    print(f"✅ {i:2d}/{len(image_files)}: {final_filename} (перезаписан)")
                else:
                    print(f"✅ {i:2d}/{len(image_files)}: {final_filename}")
            else:
                stats['failed'] += 1
                print(f"❌ {i:2d}/{len(image_files)}: {final_filename}")
        
        print(f"\n📊 Готово! Успешно: {stats['processed']}/{stats['total']}")
        return stats

    def _normalize_filename(self, filename):
        """
        Нормализует имя файла, оставляя только разрешенные символы
        """
        path = Path(filename)
        name_without_ext = path.stem
        original_ext = path.suffix
        
        print(f"Нормализация имени: '{name_without_ext}'")
        
        # Разрешенные символы: русские и английские буквы, цифры, основные спецсимволы
        allowed_chars = set(
            'abcdefghijklmnopqrstuvwxyz' +
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ' +
            'абвгдеёжзийклмнопрстуфхцчшщъыьэюя' +
            'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ' +
            '0123456789' +
            '_- .()[]{}!@#$%^&+='
        )
        
        # Нормализуем имя файла
        normalized_name = ''
        for char in name_without_ext:
            if char in allowed_chars:
                normalized_name += char
            else:
                # Заменяем запрещенные символы на подчеркивание
                print(f"Заменяем символ: '{char}' (код: {ord(char)})")
                normalized_name += '_'
        
        # Если после нормализации имя пустое, создаем случайное
        if not normalized_name.strip('_. '):
            import random
            import string
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            normalized_name = f"image_{random_suffix}"
        
        # Убираем множественные подчеркивания и обрезаем длинные имена
        normalized_name = '_'.join(filter(None, normalized_name.split('_')))
        if len(normalized_name) > 100:
            normalized_name = normalized_name[:100]
        
        result = normalized_name + original_ext
        print(f"После нормализации имени: '{result}'")
        return filename

    def _normalize_extension(self, filename):
        """
        Нормализует расширение файла, оставляя только одно стандартное расширение
        """
        path = Path(filename)
        name_without_ext = path.stem
        
        # Получаем все возможные расширения (могут быть множественные)
        suffixes = path.suffixes
        original_ext = ''.join(suffixes).lower() if suffixes else ''
        
        print(f"Нормализация расширения: '{filename}' -> имя: '{name_without_ext}', расширения: {suffixes}")
        
        # Словарь для нормализации расширений
        extension_map = {
            '.jpeg': '.jpg', '.jpe': '.jpg', '.jfif': '.jpg', '.jif': '.jpg',
            '.ipg': '.jpg', '.jpgg': '.jpg', '.jpg.jpg': '.jpg',
            '.png': '.png', '.tiff': '.tiff', '.tif': '.tiff',
            '.bmp': '.bmp', '.gif': '.gif', '.webp': '.webp'
        }
        
        # Нормализуем комбинацию расширений
        combined_ext = original_ext
        normalized_ext = extension_map.get(combined_ext, '.jpg')  # по умолчанию jpg
        
        # Если комбинированного расширения нет в маппинге, берем последнее
        if combined_ext not in extension_map and suffixes:
            last_ext = suffixes[-1].lower()
            normalized_ext = extension_map.get(last_ext, '.jpg')
        
        # Убираем все существующие расширения из имени
        final_name = name_without_ext
        while True:
            temp_path = Path(final_name)
            temp_suffix = temp_path.suffix.lower()
            if temp_suffix and (temp_suffix in extension_map or 
                               any(ext in temp_suffix for ext in ['.jpg', '.png', '.tiff', '.bmp', '.gif', '.webp'])):
                final_name = temp_path.stem
            else:
                break
        
        # Убираем лишние символы в конце имени
        final_name = final_name.rstrip('«»„"”´`¨¯¸ºª¿¡')
        
        result = final_name + normalized_ext
        print(f"После нормализации расширения: '{result}'")
        return result

    def _decode_corrupted_filename(self, filename):
        """
        Пытается декодировать испорченные имена файлов с русскими буквами
        """
        try:
            # Убираем лишние расширения сначала
            clean_name = Path(filename).stem
            
            print(f"Декодируем имя: '{clean_name}'")
            
            # Пробуем прямое исправление символов
            fixed_by_mapping = self._fix_double_encoding(clean_name)
            if fixed_by_mapping != clean_name:
                print(f"Исправлено по маппингу: '{clean_name}' -> '{fixed_by_mapping}'")
                return fixed_by_mapping
            
            # Пробуем исправить двойную перекодировку
            try:
                # ÉâÇäÇ в UTF-8 байтах -> декодируем как Windows-1251
                current_bytes = clean_name.encode('utf-8')
                decoded_name = current_bytes.decode('windows-1251')
                
                print(f"Двойное декодирование: '{clean_name}' -> '{decoded_name}'")
                
                if any(cyrillic in decoded_name for cyrillic in 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'):
                    return decoded_name
                    
            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                print(f"Ошибка двойного декодирования: {e}")
            
            print(f"Декодирование не изменило имя, возвращаем: '{clean_name}'")
            return clean_name
            
        except Exception as e:
            print(f"Ошибка декодирования {filename}: {e}")
            return filename

    def _fix_double_encoding(self, corrupted_name):
        """
        Специальная функция для исправления двойной перекодировки
        Пример: 'РГАДА' -> 'ÉâÇäÇ'
        """
        
        mapping = {
            # Заглавные русские буквы
            'É': 'Р', 'â': 'Г', 'Ç': 'А', 'ä': 'Д', 
            'à': 'Б', 'á': 'В', 'ã': 'Г', 'å': 'Е', 'ç': 'З',
            'è': 'И', 'é': 'Й', 'ê': 'К', 'ë': 'Л', 'ì': 'М',
            'í': 'Б', 'î': 'О', 'ï': 'П', 'ð': 'Р', 'ñ': 'С',
            'ó': 'У', 'ô': 'Ф', 'õ': 'Х', 'ö': 'Ц', '÷': 'Ч',
            'ø': 'Ш', 'ù': 'Щ', 'ú': 'Ъ', 'û': 'Ы', 'ü': 'Ь',
            'ý': 'Э', 'þ': 'Ю', 'ÿ': 'Я',
            
            # Специальные символы
            '«': 'О', '»': '-', '„': '-', '“': '-', '”': '-',
            '´': "'", '`': "'", '¨': '"', '¯': '-', '¸': ',',
            'º': '.', 'ª': '.', '¿': '?', '¡': '!',  # для вашего случая с «í
        }
        
        fixed_name = ''
        for char in corrupted_name:
            if char in mapping:
                fixed_name += mapping[char]
            else:
                fixed_name += char
        
        return corrupted_name

