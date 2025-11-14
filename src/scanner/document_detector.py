import cv2
import numpy as np
from typing import Optional, Tuple, List
from src.utils.config import ProcessingConfig

class DocumentDetector:
    """Класс для автоматического обнаружения документов на изображениях"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
    
    def detect_document_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Обнаруживает контур документа на изображении.
        Возвращает контур с 4 вершинами или None если не найден.
        """
        # Конвертация в grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Предобработка
        blurred = cv2.GaussianBlur(gray, self.config.gaussian_blur_kernel, 0)
        edged = cv2.Canny(blurred, self.config.canny_threshold1, self.config.canny_threshold2)
        
        # Поиск контуров
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Сортировка по площади (от большего к меньшему)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # Поиск четырехугольного контура
        for contour in contours:
            epsilon = self.config.contour_approximation_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return approx
        
        return None
    
    def preprocess_for_detection(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для улучшения детекции"""
        # Можно добавить дополнительные шаги: 
        # повышение контраста, морфологические операции и т.д.
        return image
