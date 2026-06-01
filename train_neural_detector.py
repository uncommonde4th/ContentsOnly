#!/usr/bin/env python3
"""
Обучение детектора документов на основе YOLOv8-pose
Исправленная версия
"""

import sys
from pathlib import Path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение детектора документов')
    parser.add_argument('--data', type=str, required=True,
                       help='Путь к dataset.yaml')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох обучения')
    parser.add_argument('--batch', type=int, default=4,
                       help='Размер батча')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Размер входного изображения')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Устройство: cpu или cuda')
    parser.add_argument('--model', type=str, default='models/best.pt',
                       help='Базовая модель')
    
    args = parser.parse_args()
    
    # Проверяем наличие ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ Ошибка: ultralytics не установлен")
        print("   Установите: pip install ultralytics")
        sys.exit(1)
    
    # Проверяем существование dataset.yaml
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Файл не найден: {args.data}")
        print("   Сначала запустите prepare_dataset.py")
        sys.exit(1)
    
    print(f"🚀 Начинаем обучение...")
    print(f"   Данные: {args.data}")
    print(f"   Эпохи: {args.epochs}")
    print(f"   Батч: {args.batch}")
    print(f"   Размер: {args.img_size}")
    print(f"   Устройство: {args.device}")
    print(f"   Модель: {args.model}")
    
    # Загружаем предобученную модель YOLOv8-pose
    print(f"\n📥 Загрузка предобученной модели { args.model }...")
    model = YOLO(args.model)
    
    # Параметры обучения
    # Убираем kpt_shape и другие параметры, которые не поддерживаются
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        workers=6,  # Для отладки
        project='models',
        name='doc_detector',
        exist_ok=True,
        verbose=True,
        patience=20,  # Ранняя остановка если нет улучшений
        save=True,
        save_period=10,
        cache=True,
        # Аугментация данных (важно для маленького датасета)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
    )
    
    # Экспортируем в ONNX для быстрого инференса
    print("\n📦 Экспортируем в ONNX...")
    model.export(format='onnx', imgsz=args.img_size)
    
    # Копируем лучшую модель в корень
    best_model = Path('models/doc_detector/weights/best.pt')
    if best_model.exists():
        import shutil
        shutil.copy(best_model, 'models/doc_detector.pt')
        print(f"\n✅ Модель сохранена: models/doc_detector.pt")
    else:
        print(f"\n⚠️ Модель не найдена в {best_model}")
    
    print("\n✨ Обучение завершено!")

if __name__ == '__main__':
    main()