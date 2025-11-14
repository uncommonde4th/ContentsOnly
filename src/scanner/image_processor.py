import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Optional
from src.scanner.document_detector import DocumentDetector
from src.scanner.perspective_transform import PerspectiveTransformer
from src.utils.config import ProcessingConfig

class ImageProcessor:
    """Основной класс для обработки изображений"""
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.detector = DocumentDetector(config)
        self.transformer = PerspectiveTransformer()
    
    def process_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Обрабатывает одно изображение: обнаруживает документ и обрезает его.
        Возвращает обрезанное изображение или None при ошибке.
        """
        try:
            # Загрузка изображения
            image = cv2.imread(image_path)
            if image is None:
                print(f"Ошибка загрузки изображения: {image_path}")
                return None
            
            # Обнаружение контура документа
            contour = self.detector.detect_document_contour(image)
            
            if contour is not None:
                # Выравнивание перспективы
                if self.config.enable_perspective_correction:
                    result = self.transformer.four_point_transform(
                        image, contour.reshape(4, 2), self.config.margin_pixels
                    )
                else:
                    # Простая обрезка по bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    result = image[y:y+h, x:x+w]
                
                return result
            else:
                print(f"Документ не обнаружен на изображении: {image_path}")
                return image  # Возвращаем оригинал если документ не найден
                
        except Exception as e:
            print(f"Ошибка обработки {image_path}: {str(e)}")
            return None
    
    def process_folder(self, input_folder: str, output_folder: str) -> dict:
        """
        Обрабатывает все JPEG изображения в папке.
        Возвращает статистику обработки.
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Поиск JPEG файлов
        image_extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG']
        image_files = []
        for extension in image_extensions:
            image_files.extend(input_path.glob(extension))
        
        stats = {
            'total': len(image_files),
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for image_file in image_files:
            output_file = output_path / f"cropped_{image_file.name}"
            
            result_image = self.process_single_image(str(image_file))
            
            if result_image is not None:
                # Сохранение результата
                if self.config.output_format.upper() == "JPEG":
                    cv2.imwrite(
                        str(output_file), 
                        result_image, 
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
                    )
                else:
                    cv2.imwrite(str(output_file), result_image)
                
                stats['processed'] += 1
            else:
                stats['failed'] += 1
        
        return stats
