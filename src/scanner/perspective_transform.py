import cv2
import numpy as np
from typing import Tuple, List

class PerspectiveTransformer:
    """Класс для выравнивания перспективы документа"""
    
    @staticmethod
    def order_points(pts: np.ndarray) -> np.ndarray:
        """
        Упорядочивает точки в порядке:
        верх-лево, верх-право, низ-право, низ-лево
        """
        rect = np.zeros((4, 2), dtype="float32")
        
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # верх-лево имеет наименьшую сумму
        rect[2] = pts[np.argmax(s)]  # низ-право имеет наибольшую сумму
        
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # верх-право имеет наименьшую разность
        rect[3] = pts[np.argmax(diff)]  # низ-лево имеет наибольшую разность
        
        return rect
    
    def four_point_transform(self, image: np.ndarray, pts: np.ndarray, 
                           margin: int = 0) -> np.ndarray:
        """
        Применяет преобразование перспективы к четырехугольной области
        """
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        
        # Вычисляем ширину новой картинки
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))
        
        # Вычисляем высоту новой картинки
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))
        
        # Добавляем отступы
        maxWidth += 2 * margin
        maxHeight += 2 * margin
        
        # Формируем набор точек для "вида сверху"
        dst = np.array([
            [margin, margin],
            [maxWidth - 1 - margin, margin],
            [maxWidth - 1 - margin, maxHeight - 1 - margin],
            [margin, maxHeight - 1 - margin]], dtype="float32")
        
        # Вычисляем матрицу преобразования и применяем ее
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped
