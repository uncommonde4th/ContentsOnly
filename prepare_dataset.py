#!/usr/bin/env python3
"""
Конвертирует LabelMe аннотации в формат YOLOv8-pose (17 колонок)
Формат: class_id x_center y_center width height x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4
"""

import json
import cv2
import shutil
from pathlib import Path
import random
import numpy as np

def convert_labelme_to_yolo(annotations_folder, images_folder, output_folder, train_ratio=0.8):
    """
    Конвертирует аннотации из LabelMe в YOLOv8-pose формат (17 колонок)
    """
    
    output_path = Path(output_folder)
    
    # Создаём структуру папок
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Собираем все JSON файлы
    json_files = list(Path(annotations_folder).glob('*.json'))
    
    if not json_files:
        print(f"❌ Не найдено JSON файлов в {annotations_folder}")
        return
    
    print(f"📁 Найдено {len(json_files)} аннотаций")
    
    # Перемешиваем и разделяем на train/val
    random.shuffle(json_files)
    split_idx = int(len(json_files) * train_ratio)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]
    
    print(f"📊 Train: {len(train_files)} изображений")
    print(f"📊 Val: {len(val_files)} изображений")
    
    # Конвертируем train
    print("\n🔄 Конвертация train набора...")
    for json_file in train_files:
        convert_single_annotation(json_file, images_folder, 
                                  output_path / 'images' / 'train', 
                                  output_path / 'labels' / 'train')
    
    # Конвертируем val
    print("\n🔄 Конвертация val набора...")
    for json_file in val_files:
        convert_single_annotation(json_file, images_folder,
                                  output_path / 'images' / 'val',
                                  output_path / 'labels' / 'val')
    
    # Создаём dataset.yaml
    create_dataset_yaml(output_path, len(train_files), len(val_files))
    
    print(f"\n✅ Готово!")
    print(f"   Датасет сохранён в: {output_path}")
    print(f"   Train: {len(train_files)} изображений")
    print(f"   Val: {len(val_files)} изображений")

def convert_single_annotation(json_file, images_folder, images_output, labels_output):
    """Конвертирует один JSON файл в формат YOLOv8-pose (17 колонок)"""
    
    # Загружаем JSON
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Получаем имя изображения
    img_name = data.get('imagePath', '')
    img_width = data.get('imageWidth', 0)
    img_height = data.get('imageHeight', 0)
    
    if not img_name or img_width == 0 or img_height == 0:
        print(f"   ⚠️ Неверные данные в {json_file.name}")
        return
    
    # Ищем изображение в папке
    img_path = None
    possible_names = [
        Path(images_folder) / img_name,
        Path(images_folder) / f"{Path(img_name).stem}.jpg",
        Path(images_folder) / f"{Path(img_name).stem}.jpeg",
        Path(images_folder) / f"{Path(img_name).stem}.png",
        Path(images_folder) / f"{Path(img_name).stem}.JPG",
        Path(images_folder) / f"{Path(img_name).stem}.JPEG",
        Path(images_folder) / f"{Path(img_name).stem}.PNG",
    ]
    
    for test_path in possible_names:
        if test_path.exists():
            img_path = test_path
            break
    
    if img_path is None:
        print(f"   ⚠️ Изображение не найдено: {img_name}")
        return
    
    # Получаем точки из аннотации
    points = None
    for shape in data.get('shapes', []):
        if shape.get('label') == 'document' and shape.get('shape_type') == 'polygon':
            points = shape.get('points', [])
            break
    
    if not points or len(points) != 4:
        print(f"   ⚠️ Неверное количество точек в {json_file.name}: {len(points) if points else 0}")
        return
    
    # Упорядочиваем точки: TL, TR, BR, BL (top-left, top-right, bottom-right, bottom-left)
    ordered_points = order_points_clockwise(points)
    
    # Вычисляем bounding box
    xs = [p[0] for p in ordered_points]
    ys = [p[1] for p in ordered_points]
    
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    
    # Центр и размеры bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Нормализуем координаты (0-1)
    norm_x_center = x_center / img_width
    norm_y_center = y_center / img_height
    norm_width = width / img_width
    norm_height = height / img_height
    
    # Формируем YOLOv8-pose аннотацию (17 колонок)
    # Формат: class x_center y_center width height x1 y1 vis1 x2 y2 vis2 x3 y3 vis3 x4 y4 vis4
    # visibility: 0 = не видна, 1 = закрыта/частично, 2 = полностью видна
    class_id = 0
    visibility = 2  # Все точки считаем полностью видимыми
    
    parts = [
        str(class_id),
        f"{norm_x_center:.6f}",
        f"{norm_y_center:.6f}",
        f"{norm_width:.6f}",
        f"{norm_height:.6f}"
    ]
    
    # Добавляем 4 ключевые точки
    for x, y in ordered_points:
        norm_x = max(0.0, min(1.0, x / img_width))
        norm_y = max(0.0, min(1.0, y / img_height))
        parts.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
    
    # Проверяем количество колонок
    if len(parts) != 13:
        print(f"   ❌ Ошибка: ожидалось 13 колонок, получено {len(parts)}")
        return
    
    # Сохраняем YOLO аннотацию
    label_file = labels_output / f"{Path(img_name).stem}.txt"
    with open(label_file, 'w') as f:
        f.write(" ".join(parts))
    
    # Копируем изображение
    shutil.copy2(img_path, images_output / img_path.name)
    
    print(f"   ✅ {img_name} -> {label_file.name} (13 колонок)")

def order_points_clockwise(points):
    """
    Упорядочивает 4 точки по часовой стрелке, начиная с верхнего-левого угла
    Порядок: TL (0), TR (1), BR (2), BL (3)
    """
    pts = np.array(points, dtype=np.float32)
    
    # Сортируем по y координате (верхние точки сначала)
    y_sorted = pts[np.argsort(pts[:, 1])]
    
    # Разделяем на верхние и нижние
    top_pts = y_sorted[:2]
    bottom_pts = y_sorted[2:]
    
    # Сортируем верхние по x (слева направо)
    top_left = top_pts[np.argmin(top_pts[:, 0])]
    top_right = top_pts[np.argmax(top_pts[:, 0])]
    
    # Сортируем нижние по x (слева направо)
    bottom_left = bottom_pts[np.argmin(bottom_pts[:, 0])]
    bottom_right = bottom_pts[np.argmax(bottom_pts[:, 0])]
    
    return [
        (float(top_left[0]), float(top_left[1])),   # 0: TL
        (float(top_right[0]), float(top_right[1])), # 1: TR
        (float(bottom_right[0]), float(bottom_right[1])), # 2: BR
        (float(bottom_left[0]), float(bottom_left[1]))    # 3: BL
    ]

def create_dataset_yaml(output_path, train_count, val_count):
    """Создаёт dataset.yaml для YOLOv8-pose"""
    
    yaml_content = f"""# YOLOv8-pose dataset configuration
# Train: {train_count} images, Val: {val_count} images

path: {output_path.absolute()}
train: images/train
val: images/val

nc: 1  # number of classes
names: ['document']  # class names

# Keypoints configuration (4 corners of document)
kpt_shape: [4, 2]  # 4 keypoints, each with x,y
flip_idx: [1, 0, 3, 2]  # При горизонтальном отражении: TL↔TR, BR↔BL
"""
    
    (output_path / 'dataset.yaml').write_text(yaml_content)
    print(f"\n📄 Создан dataset.yaml")

def visualize_yolo_annotations(labels_folder, images_folder, output_folder=None):
    """
    Визуализирует YOLO аннотации для проверки
    """
    output_folder = Path(output_folder) if output_folder else Path(labels_folder).parent / 'visualization'
    output_folder.mkdir(parents=True, exist_ok=True)
    
    labels_path = Path(labels_folder)
    
    for split in ['train', 'val']:
        split_path = labels_path / split
        if not split_path.exists():
            continue
        
        txt_files = list(split_path.glob('*.txt'))
        img_folder = Path(images_folder) / split
        
        print(f"\n🔍 Визуализация {split}: {len(txt_files)} аннотаций...")
        
        for txt_file in txt_files[:5]:  # Показываем первые 5
            # Загружаем аннотацию
            with open(txt_file, 'r') as f:
                line = f.readline().strip()
            
            parts = line.split()
            if len(parts) != 17:
                print(f"   ⚠️ Неверный формат: {txt_file.name} ({len(parts)} колонок)")
                continue
            
            # Загружаем изображение
            img_name = txt_file.stem
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_path = img_folder / f"{img_name}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
            
            if img_path is None:
                print(f"   ⚠️ Изображение не найдено: {img_name}")
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ⚠️ Не удалось загрузить: {img_path}")
                continue
            
            h, w = img.shape[:2]
            
            # Парсим аннотацию
            # class x_center y_center width height x1 y1 vis1 ...
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            
            # Рисуем bbox
            x_min = int(x_center - width/2)
            y_min = int(y_center - height/2)
            x_max = int(x_center + width/2)
            y_max = int(y_center + height/2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Рисуем ключевые точки
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for i in range(4):
                idx = 5 + i * 3
                x = float(parts[idx]) * w
                y = float(parts[idx + 1]) * h
                vis = int(parts[idx + 2])
                
                cv2.circle(img, (int(x), int(y)), 8, colors[i], -1)
                cv2.putText(img, str(i), (int(x)+10, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)
                
                # Соединяем точки линией (полигон)
                next_idx = 5 + ((i + 1) % 4) * 3
                x2 = float(parts[next_idx]) * w
                y2 = float(parts[next_idx + 1]) * h
                cv2.line(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 255), 2)
            
            # Сохраняем
            output_path = output_folder / f"viz_{split}_{txt_file.name}.jpg"
            cv2.imwrite(str(output_path), img)
            print(f"   ✅ {output_path}")
    
    print(f"\n✨ Визуализация завершена! Результаты в: {output_folder}")

def fix_existing_annotations(labels_folder):
    """
    Исправляет существующие YOLO аннотации (17 колонок -> 13 колонок)
    Убирает visibility из каждой точки: x y vis -> x y
    """
    labels_path = Path(labels_folder)

    for split in ['train', 'val']:
        split_path = labels_path / split
        if not split_path.exists():
            continue

        txt_files = list(split_path.glob('*.txt'))
        fixed_count = 0

        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                line = f.readline().strip()

            parts = line.split()

            if len(parts) == 17:
                # class + bbox (5) + 4 точки по x y vis (12) -> убираем vis
                new_parts = parts[:5]  # class x_c y_c w h
                for i in range(4):
                    idx = 5 + i * 3
                    new_parts += [parts[idx], parts[idx + 1]]  # только x y, без vis

                with open(txt_file, 'w') as f:
                    f.write(" ".join(new_parts))

                fixed_count += 1

            elif len(parts) == 13:
                pass  # Уже правильный формат
            else:
                print(f"   ⚠️ Неизвестный формат в {txt_file.name}: {len(parts)} колонок")

        if fixed_count > 0:
            print(f"   ✅ Исправлено {fixed_count} файлов в {split}")

    print(f"\n✨ Исправление завершено!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Подготовка датасета для YOLOv8-pose (17 колонок)')
    parser.add_argument('--annotations', type=str, 
                       default='./labeled_data/annotations/labelme',
                       help='Папка с JSON аннотациями')
    parser.add_argument('--images', type=str,
                       default='./labeled_data/images',
                       help='Папка с исходными изображениями')
    parser.add_argument('--output', type=str,
                       default='./dataset',
                       help='Папка для сохранения YOLO датасета')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Доля изображений для обучения (0-1)')
    parser.add_argument('--visualize', type=str,
                       help='Визуализировать YOLO аннотации (указать папку labels)')
    parser.add_argument('--fix', type=str,
                       help='Исправить существующие YOLO аннотации (13 -> 17 колонок)')
    
    args = parser.parse_args()
    
    if args.fix:
        print("🔧 Исправление существующих аннотаций (13 -> 17 колонок)...")
        fix_existing_annotations(args.fix)
    elif args.visualize:
        print("🔍 Визуализация YOLO аннотаций...")
        visualize_yolo_annotations(args.visualize, Path(args.visualize).parent / 'images')
    else:
        print("🚀 Начинаем конвертацию датасета в YOLOv8-pose формат (17 колонок)...")
        convert_labelme_to_yolo(args.annotations, args.images, args.output, args.train_ratio)