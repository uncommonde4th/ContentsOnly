import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

class TextDocumentDetector:
    """Специализированный детектор для текстовых документов на белом фоне"""
    
    def __init__(self):
        self.min_text_area_ratio = 0.3  # Минимум 30% текстовой области
        self.margin_ratio = 0.05  # 5% отступ
    
    def detect_text_regions(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Обнаруживает текстовые регионы и возвращает bounding box (x, y, w, h)
        """
        # Конвертируем в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Метод 1: Поиск текста через адаптивный порог
        binary1 = self._adaptive_threshold_method(gray)
        
        # Метод 2: Поиск текста через детектор краев
        binary2 = self._edge_based_method(gray)
        
        # Метод 3: Поиск текста через морфологические операции
        binary3 = self._morphological_method(gray)
        
        # Комбинируем все методы
        combined = cv2.bitwise_or(binary1, binary2)
        combined = cv2.bitwise_or(combined, binary3)
        
        # Улучшаем маску
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Находим контуры текстовых блоков
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Объединяем все текстовые блоки в один bounding box
        all_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Игнорируем очень маленькие контуры
                x, y, w, h = cv2.boundingRect(contour)
                all_points.extend([(x, y), (x + w, y + h)])
        
        if not all_points:
            return None
        
        # Находим общий bounding box всех текстовых областей
        all_points = np.array(all_points)
        x_min = np.min(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        x_max = np.max(all_points[:, 0])
        y_max = np.max(all_points[:, 1])
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _adaptive_threshold_method(self, gray: np.ndarray) -> np.ndarray:
        """Метод адаптивного порога для текста"""
        # Адаптивный порог для текста
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 5
        )
        
        # Удаляем очень маленькие объекты (шум)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary
    
    def _edge_based_method(self, gray: np.ndarray) -> np.ndarray:
        """Метод на основе краев для текста"""
        # Детектор краев для текста
        edges = cv2.Canny(gray, 50, 150)
        
        # Морфологические операции для соединения текстовых линий
        kernel = np.ones((2, 1), np.uint8)  # Вертикальное ядро для текста
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def _morphological_method(self, gray: np.ndarray) -> np.ndarray:
        """Морфологический метод для текста"""
        # Создаем маску для текста через градиенты
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient = cv2.magnitude(grad_x, grad_y)
        gradient = np.uint8(255 * gradient / np.max(gradient))
        
        # Порог для текстовых областей
        _, text_mask = cv2.threshold(gradient, 30, 255, cv2.THRESH_BINARY)
        
        return text_mask
    
    def detect_text_document(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Основной метод обнаружения текстового документа
        """
        h, w = image.shape[:2]
        image_area = h * w
        
        print("🔤 Поиск текстового документа...")
        
        # Находим текстовые регионы
        bbox = self.detect_text_regions(image)
        
        if bbox is None:
            print("❌ Текстовые регионы не найдены")
            return None
        
        x, y, text_w, text_h = bbox
        text_area = text_w * text_h
        text_area_ratio = text_area / image_area
        
        print(f"📄 Найден текст: {text_w}x{text_h} ({text_area_ratio*100:.1f}% изображения)")
        
        # Проверяем что текст занимает значительную область
        if text_area_ratio < self.min_text_area_ratio:
            print(f"⚠️  Текстовая область слишком маленькая ({text_area_ratio*100:.1f}%)")
            return None
        
        # Добавляем отступы вокруг текста
        margin_x = int(text_w * self.margin_ratio)
        margin_y = int(text_h * self.margin_ratio)
        
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(w, x + text_w + margin_x)
        y2 = min(h, y + text_h + margin_y)
        
        # Создаем контур документа (прямоугольник вокруг текста)
        document_contour = np.array([
            [x1, y1],
            [x2, y1], 
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
        
        print(f"✅ Текстовый документ обнаружен: {x2-x1}x{y2-y1}")
        return document_contour
    
    def detect_with_page_borders(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Метод поиска границ страницы через анализ гистограмм
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Анализ гистограмм по краям для поиска границ страницы
        left_border = self._find_page_border(gray, 'left')
        right_border = self._find_page_border(gray, 'right')
        top_border = self._find_page_border(gray, 'top')
        bottom_border = self._find_page_border(gray, 'bottom')
        
        if all(border is not None for border in [left_border, right_border, top_border, bottom_border]):
            # Создаем контур из найденных границ
            contour = np.array([
                [left_border, top_border],
                [right_border, top_border],
                [right_border, bottom_border],
                [left_border, bottom_border]
            ], dtype=np.int32)
            
            print(f"📄 Границы страницы найдены: {right_border-left_border}x{bottom_border-top_border}")
            return contour
        
        return None
    
    def _find_page_border(self, gray: np.ndarray, side: str) -> Optional[int]:
        """Находит границу страницы анализируя гистограмму"""
        h, w = gray.shape
        
        if side == 'left':
            strip = gray[:, :50]  # Левая полоса 50px
            hist = np.mean(strip, axis=0)
            changes = np.diff(hist > np.mean(hist) * 1.1)
            borders = np.where(changes)[0]
            return borders[0] if len(borders) > 0 else None
            
        elif side == 'right':
            strip = gray[:, -50:]  # Правая полоса 50px
            hist = np.mean(strip, axis=0)
            changes = np.diff(hist > np.mean(hist) * 1.1)
            borders = np.where(changes)[0]
            return w - 50 + borders[-1] if len(borders) > 0 else None
            
        elif side == 'top':
            strip = gray[:50, :]  # Верхняя полоса 50px
            hist = np.mean(strip, axis=1)
            changes = np.diff(hist > np.mean(hist) * 1.1)
            borders = np.where(changes)[0]
            return borders[0] if len(borders) > 0 else None
            
        elif side == 'bottom':
            strip = gray[-50:, :]  # Нижняя полоса 50px
            hist = np.mean(strip, axis=1)
            changes = np.diff(hist > np.mean(hist) * 1.1)
            borders = np.where(changes)[0]
            return h - 50 + borders[-1] if len(borders) > 0 else None
        
        return None
