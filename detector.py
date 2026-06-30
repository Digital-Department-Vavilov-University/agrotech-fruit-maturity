"""
detector.py — детектор зрелости продукции на основе YOLOv8.

Режимы работы:
  python detector.py --image path/to/image.jpg --weights results/maturity_detector/weights/best.pt
  python detector.py --camera 0               --weights results/...
  python detector.py --demo                   (демо на синтетических данных без GPU)
"""

import argparse
import sys
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np

# ──────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────
CLASSES = ["unripe", "semi_ripe", "ripe", "overripe"]

CLASS_COLORS_BGR = {
    "unripe":    (50,  180,  50),   # зелёный
    "semi_ripe": (30,  190, 230),   # жёлто-оранжевый
    "ripe":      (50,   80, 220),   # красный
    "overripe":  (30,   30, 130),   # тёмно-красный
}

# Рекомендация для АПК-робота
ROBOT_ACTIONS = {
    "unripe":    "ПРОПУСТИТЬ — не готов к уборке",
    "semi_ripe": "ОТМЕТИТЬ — уборка через 3–5 дней",
    "ripe":      "ЗАХВАТ — немедленная уборка",
    "overripe":  "УТИЛИЗАЦИЯ — нарушение стандарта",
}


# ──────────────────────────────────────────────────
# Структура данных
# ──────────────────────────────────────────────────
@dataclass
class MaturityDetection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    center:     Tuple[int, int]
    action:     str = field(init=False)

    def __post_init__(self):
        self.action = ROBOT_ACTIONS[self.class_name]


# ──────────────────────────────────────────────────
# Детектор
# ──────────────────────────────────────────────────
class MaturityDetector:
    def __init__(self, weights: str, conf_threshold: float = 0.45, iou_threshold: float = 0.50):
        """
        weights       — путь к файлу best.pt (YOLOv8)
        conf_threshold — порог уверенности детекции
        iou_threshold  — порог NMS IoU
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("Установите ultralytics: pip install ultralytics")

        self.model = YOLO(weights)
        self.conf  = conf_threshold
        self.iou   = iou_threshold

    def detect(self, image: np.ndarray) -> List[MaturityDetection]:
        """Запускает инференс, возвращает список MaturityDetection."""
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(MaturityDetection(
                    class_id=cls_id,
                    class_name=CLASSES[cls_id],
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                ))
        # Сортировка: сначала зрелые (приоритет для захвата)
        detections.sort(key=lambda d: d.class_id == 2, reverse=True)
        return detections

    @staticmethod
    def visualize(image: np.ndarray, detections: List[MaturityDetection]) -> np.ndarray:
        """Накладывает аннотации на изображение."""
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS_BGR.get(det.class_name, (200, 200, 200))

            # Bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Центроид
            cv2.circle(vis, det.center, 5, color, -1, cv2.LINE_AA)

            # Метка
            label = f"{det.class_name}  {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
            cv2.putText(vis, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)

        # Панель статуса (для интеграции с АПК-роботом)
        ripe_count = sum(1 for d in detections if d.class_name == "ripe")
        status = f"Объектов: {len(detections)}  |  К уборке: {ripe_count}"
        cv2.rectangle(vis, (0, 0), (len(status) * 9 + 10, 22), (0, 0, 0), -1)
        cv2.putText(vis, status, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 200), 1, cv2.LINE_AA)
        return vis


# ──────────────────────────────────────────────────
# Вспомогательный демо-режим (без обученной модели)
# ──────────────────────────────────────────────────
def run_demo():
    """Запускает демонстрацию на синтетических изображениях (без GPU/весов)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_dataset import generate_background, place_objects, IMG_W, IMG_H, CLASSES as GEN_CLASSES

    rng = np.random.default_rng(0)
    print("\nДемонстрационный режим (синтетические данные, без YOLOv8)\n")

    for sample_idx in range(5):
        img = generate_background(rng)
        annots = place_objects(img, rng)

        # Имитируем «предсказание» из аннотаций (для демо-визуализации)
        detections = []
        for cls_id, cx_n, cy_n, w_n, h_n in annots:
            cx = int(cx_n * IMG_W)
            cy = int(cy_n * IMG_H)
            w  = int(w_n  * IMG_W)
            h  = int(h_n  * IMG_H)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            detections.append(MaturityDetection(
                class_id=cls_id,
                class_name=GEN_CLASSES[cls_id],
                confidence=round(0.70 + rng.random() * 0.29, 2),
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
            ))

        vis = MaturityDetector.visualize(img, detections)
        out_path = f"demo_sample_{sample_idx:02d}.jpg"
        cv2.imwrite(out_path, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  Образец {sample_idx}: {len(detections)} объектов → {out_path}")
        for d in detections:
            print(f"    [{d.class_name:10s}] conf={d.confidence:.2f}  действие: {d.action}")

    print("\nДемо завершено. Файлы demo_sample_*.jpg сохранены.")


# ──────────────────────────────────────────────────
# Режимы: изображение / камера
# ──────────────────────────────────────────────────
def run_image(weights, image_path, conf, iou):
    detector = MaturityDetector(weights, conf, iou)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        sys.exit(1)

    t0 = time.perf_counter()
    detections = detector.detect(img)
    ms = (time.perf_counter() - t0) * 1000

    vis = MaturityDetector.visualize(img, detections)
    out = image_path.replace(".", "_detected.", 1)
    cv2.imwrite(out, vis)

    print(f"\nИзображение: {image_path}")
    print(f"Время инференса: {ms:.1f} мс  |  Объектов: {len(detections)}")
    for d in detections:
        print(f"  [{d.class_name:10s}] conf={d.confidence:.2f}  bbox={d.bbox}  → {d.action}")
    print(f"Результат сохранён: {out}")


def run_camera(weights, camera_id, conf, iou):
    detector = MaturityDetector(weights, conf, iou)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть камеру {camera_id}")
        sys.exit(1)

    print(f"Камера {camera_id} запущена. Нажмите 'q' для выхода.")
    fps_buf = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections = detector.detect(frame)
        ms = (time.perf_counter() - t0) * 1000
        fps_buf.append(1000 / ms if ms > 0 else 0)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = sum(fps_buf) / len(fps_buf)

        vis = MaturityDetector.visualize(frame, detections)
        cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1] - 120, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow("Maturity Detector — APC Robot", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────
# Точка входа
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Детектор зрелости YOLOv8 для АПК-робота")
    parser.add_argument("--weights", default="results/maturity_detector/weights/best.pt")
    parser.add_argument("--image",   help="Путь к изображению")
    parser.add_argument("--camera",  type=int, help="Индекс камеры (0, 1, ...)")
    parser.add_argument("--demo",    action="store_true", help="Демо без обученной модели")
    parser.add_argument("--conf",    type=float, default=0.45)
    parser.add_argument("--iou",     type=float, default=0.50)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.image:
        run_image(args.weights, args.image, args.conf, args.iou)
    elif args.camera is not None:
        run_camera(args.weights, args.camera, args.conf, args.iou)
    else:
        parser.print_help()
