"""
train.py — обучение модели YOLOv8 на датасете классификации зрелости продукции.

Использование:
  python train.py --data dataset/data.yaml --epochs 50 --imgsz 640
"""

import argparse
import os
import sys


def train(data_yaml: str, epochs: int, imgsz: int, batch: int, model: str, project: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ошибка: библиотека ultralytics не установлена.")
        print("Выполните: pip install ultralytics")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  YOLOv8 — Обучение классификатора зрелости продукции")
    print(f"{'='*55}")
    print(f"  Модель       : {model}")
    print(f"  Датасет      : {data_yaml}")
    print(f"  Эпохи        : {epochs}")
    print(f"  Разрешение   : {imgsz}×{imgsz}")
    print(f"  Batch-size   : {batch}")
    print(f"{'='*55}\n")

    yolo = YOLO(model)

    results = yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name="maturity_detector",
        exist_ok=True,
        verbose=True,
        plots=True,           # сохраняет графики precision/recall/loss
        save=True,
        save_period=10,       # чекпоинт каждые 10 эпох
        val=True,
        patience=20,          # early stopping
    )

    best_weights = os.path.join(project, "maturity_detector", "weights", "best.pt")
    print(f"\nОбучение завершено. Лучшие веса: {best_weights}")
    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение YOLOv8 на датасете зрелости")
    parser.add_argument("--data",    default="dataset/data.yaml", help="Путь к data.yaml")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--model",   default="yolov8n.pt",        help="Базовая модель YOLOv8")
    parser.add_argument("--project", default="results",            help="Папка результатов")
    args = parser.parse_args()

    train(args.data, args.epochs, args.imgsz, args.batch, args.model, args.project)
