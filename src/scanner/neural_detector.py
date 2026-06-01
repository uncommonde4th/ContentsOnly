"""
Модуль нейросетевого детектора документов
Использует YOLOv8-pose для обнаружения 4 углов документа
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

# Проверяем наличие ultralytics
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics не установлен. Установите: pip install ultralytics")

class NeuralDocumentDetector:
    """Нейросетевой детектор для поиска документа и его углов"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к обученной модели (.pt или .onnx)
            device: 'cpu' или 'cuda'
        """
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = device
        self.is_available = False
        
        # Путь к модели по умолчанию (если не указан)
        if model_path is None:
            # Ищем модель в стандартных местах
            possible_paths = [
                Path("models/doc_detector.pt"),
                Path("models/doc_detector.onnx"),
                Path("../models/doc_detector.pt"),
                Path("scanner/models/doc_detector.pt"),
            ]
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.logger.warning("Модель нейросети не найдена. Будет использоваться эвристический алгоритм.")
            print("⚠️ Модель нейросети не найдена. Будет использоваться эвристический алгоритм.")
            print("   Для обучения модели: python train_neural_detector.py")
    
    def load_model(self, model_path: str):
        """Загружает модель YOLO"""
        if not ULTRALYTICS_AVAILABLE:
            self.logger.error("ultralytics не установлен")
            return False
        
        try:
            self.model = YOLO(model_path)
            # Перемещаем на нужное устройство
            if self.device == 'cuda':
                self.model.to('cuda')
            self.is_available = True
            self.logger.info(f"Модель загружена: {model_path}")
            print(f"✅ Нейросетевая модель загружена: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            self.is_available = False
            return False
    
    def detect_document(self, image: np.ndarray, conf_threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Обнаруживает документ на изображении и возвращает 4 угла
        
        Args:
            image: Изображение в формате BGR (numpy array)
            conf_threshold: Порог уверенности (0-1)
        
        Returns:
            np.ndarray: 4 точки (4, 2) в порядке [top-left, top-right, bottom-right, bottom-left]
            или None если документ не найден
        """
        if not self.is_available or self.model is None:
            return None
        
        try:
            # Инференс модели
            results = self.model(image, verbose=False, conf=conf_threshold)
            
            if len(results) == 0:
                return None
            
            result = results[0]
            
            # Проверяем наличие keypoints (для YOLO-pose)
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                if len(result.keypoints.data) > 0:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    # Берем первые 4 точки (углы)
                    if len(keypoints) >= 4:
                        points = keypoints[:4, :2]  # (x, y) для каждого угла
                        # Сортируем точки в правильном порядке
                        ordered_points = self._order_points(points)
                        return ordered_points.astype(np.int32)
            
            # Проверяем наличие bounding box (если не нашли keypoints)
            if hasattr(result, 'boxes') and result.boxes is not None:
                if len(result.boxes) > 0:
                    box = result.boxes[0].xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = box.astype(np.int32)
                    # Преобразуем bounding box в 4 угла
                    points = np.array([
                        [x1, y1], [x2, y1],
                        [x2, y2], [x1, y2]
                    ], dtype=np.int32)
                    return points
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка инференса: {e}")
            return None
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Упорядочивает точки в порядке: top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Сумма координат: top-left имеет наименьшую сумму
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Разность координат: top-right имеет наименьшую разность
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def detect_with_fallback(self, image: np.ndarray, 
                            fallback_detector=None,
                            conf_threshold: float = 0.5) -> Optional[np.ndarray]:
        """
        Сначала пробует нейросеть, при неудаче — fallback детектор
        
        Args:
            image: Изображение
            fallback_detector: Функция или объект с методом find_document_auto
            conf_threshold: Порог уверенности
        
        Returns:
            4 точки или None
        """
        # Пробуем нейросеть
        points = self.detect_document(image, conf_threshold)
        
        if points is not None:
            self.logger.debug("Документ найден нейросетью")
            return points
        
        # Fallback на эвристический метод
        if fallback_detector is not None:
            self.logger.debug("Нейросеть не сработала, используем fallback")
            if hasattr(fallback_detector, 'find_document_auto'):
                contour = fallback_detector.find_document_auto(image)
                if contour is not None and len(contour) == 4:
                    return contour.reshape(4, 2)
        
        return None


# Вспомогательные функции для обучения
def prepare_training_data(image_folder: str, annotation_folder: str, output_folder: str):
    """
    Подготавливает данные для обучения YOLO
    
    Args:
        image_folder: Папка с изображениями
        annotation_folder: Папка с аннотациями в формате LabelMe (JSON)
        output_folder: Папка для сохранения подготовленных данных
    """
    import json
    from pathlib import Path
    
    output_path = Path(output_folder)
    images_path = output_path / 'images'
    labels_path = output_path / 'labels'
    
    for subdir in ['train', 'val']:
        (images_path / subdir).mkdir(parents=True, exist_ok=True)
        (labels_path / subdir).mkdir(parents=True, exist_ok=True)
    
    # Конвертируем LabelMe аннотации в формат YOLO
    json_files = list(Path(annotation_folder).glob('*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Получаем размер изображения
        img_path = Path(data['image_path'])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # Ищем polygon с 4 точками
        points = None
        for shape in data.get('shapes', []):
            if shape['shape_type'] == 'polygon' and len(shape['points']) >= 4:
                points = shape['points'][:4]
                break
        
        if points is None:
            continue
        
        # Нормализуем координаты
        norm_points = []
        for x, y in points:
            norm_points.extend([x / w, y / h])
        
        # Сохраняем в YOLO формат
        label_file = labels_path / f"{Path(img_path).stem}.txt"
        with open(label_file, 'w') as f:
            # Формат: class x1 y1 x2 y2 x3 y3 x4 y4
            f.write(f"0 " + " ".join(f"{p:.6f}" for p in norm_points))
        
        # Копируем изображение
        import shutil
        shutil.copy(img_path, images_path / img_path.name)
    
    # Создаем dataset.yaml
    yaml_content = f"""
path: {output_path.absolute()}
train: images/train
val: images/val

nc: 1
names: ['document']
nkpt: 4
kpt_shape: [4, 2]
"""
    (output_path / 'dataset.yaml').write_text(yaml_content)
    
    print(f"✅ Подготовлено {len(json_files)} изображений для обучения")
    print(f"   Сохранено в: {output_path}")
