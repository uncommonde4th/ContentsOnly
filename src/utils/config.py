from dataclasses import dataclass
from typing import Tuple

@dataclass
class ProcessingConfig:
    """Конфигурация параметров обработки изображений"""
    # Параметры детектора краев
    canny_threshold1: int = 30
    canny_threshold2: int = 150
    gaussian_blur_kernel: Tuple[int, int] = (5, 5)
    contour_approximation_epsilon: float = 0.02
    
    # Параметры сохранения
    jpeg_quality: int = 95
    output_format: str = "JPEG"  # или "PNG"
    
    # Настройки обработки
    enable_perspective_correction: bool = True
    margin_pixels: int = 5  # Отступ вокруг обрезанного документа
