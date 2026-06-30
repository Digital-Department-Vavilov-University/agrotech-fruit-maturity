"""
generate_dataset.py — генератор синтетического датасета для классификации продукции
по степени зрелости/готовности (YOLOv8, формат YOLO).

Классы зрелости (универсальные):
  0 — unripe      (незрелый)
  1 — semi_ripe   (полузрелый)
  2 — ripe        (зрелый / готовый)
  3 — overripe    (перезрелый)

Каждое изображение содержит 1–4 объекта разной степени зрелости.
Аннотации сохраняются в формате YOLO: <class> <cx> <cy> <w> <h> (нормированные).
"""

import os
import cv2
import json
import random
import argparse
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────────
# Конфигурация
# ──────────────────────────────────────────────────
CLASSES = ["unripe", "semi_ripe", "ripe", "overripe"]
NUM_CLASSES = len(CLASSES)

# HSV-диапазоны основного цвета для каждого класса зрелости
# (нижняя граница, верхняя граница, ядро цвета)
CLASS_COLOR_HSV = {
    0: {"name": "unripe",    "h": (35, 75),  "s": (120, 220), "v": (80, 160)},   # зелёный
    1: {"name": "semi_ripe", "h": (18, 35),  "s": (150, 230), "v": (100, 180)},  # жёлто-зелёный
    2: {"name": "ripe",      "h": (0, 18),   "s": (170, 255), "v": (120, 210)},  # красный/жёлтый
    3: {"name": "overripe",  "h": (0, 12),   "s": (80, 160),  "v": (40, 100)},   # тёмно-красный/бурый
}

IMG_W, IMG_H = 640, 640
SEED = 42


# ──────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────

def hsv_to_bgr(h, s, v):
    hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr[0, 0])


def random_color_for_class(cls_id, rng):
    cfg = CLASS_COLOR_HSV[cls_id]
    h = rng.integers(cfg["h"][0], cfg["h"][1])
    s = rng.integers(cfg["s"][0], cfg["s"][1])
    v = rng.integers(cfg["v"][0], cfg["v"][1])
    return hsv_to_bgr(h, s, v)


def draw_fruit(img, cx, cy, rx, ry, cls_id, rng):
    """Рисует упрощённое «плодоподобное» тело: эллипс + блик + пятна."""
    color = random_color_for_class(cls_id, rng)
    # Основное тело
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, -1, cv2.LINE_AA)

    # Объёмный градиент (несколько концентрических эллипсов светлее)
    for i in range(4):
        factor = 0.75 - i * 0.15
        light = tuple(min(255, int(c * (1.0 + 0.35 * factor))) for c in color)
        arx = max(2, int(rx * factor * 0.6))
        ary = max(2, int(ry * factor * 0.6))
        offset_x = -int(rx * 0.2)
        offset_y = -int(ry * 0.25)
        cv2.ellipse(img, (cx + offset_x, cy + offset_y), (arx, ary),
                    0, 0, 360, light, -1, cv2.LINE_AA)

    # Блик
    highlight = (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80))
    hx = cx - rx // 3
    hy = cy - ry // 3
    cv2.ellipse(img, (hx, hy), (max(2, rx // 6), max(2, ry // 6)),
                0, 0, 360, highlight, -1, cv2.LINE_AA)

    # Пятна (характерны для перезрелых)
    if cls_id == 3:
        spot_color = tuple(max(0, c - 60) for c in color)
        for _ in range(rng.integers(3, 8)):
            sx = cx + rng.integers(-rx + 5, rx - 5)
            sy = cy + rng.integers(-ry + 5, ry - 5)
            sr = rng.integers(3, 8)
            cv2.circle(img, (sx, sy), sr, spot_color, -1, cv2.LINE_AA)

    # Плодоножка
    stem_color = (30, 90, 30)
    sx = cx + rng.integers(-rx // 4, rx // 4)
    cv2.line(img, (sx, cy - ry), (sx, cy - ry - rng.integers(8, 20)), stem_color, 2, cv2.LINE_AA)


def generate_background(rng):
    """Реалистичный фон (деревянный/плетёный ящик, конвейерная лента)."""
    choice = rng.integers(0, 3)
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    if choice == 0:  # конвейерная лента — серая текстура
        base = rng.integers(80, 130, (IMG_H, IMG_W, 3), dtype=np.uint8)
        noise_small = rng.integers(0, 20, (IMG_H // 8, IMG_W // 8, 3), dtype=np.uint8)
        noise_up = cv2.resize(noise_small, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.add(base.astype(np.int16), noise_up.astype(np.int16)).clip(0, 255).astype(np.uint8)
        # полосы ленты
        for y in range(0, IMG_H, 40):
            cv2.line(img, (0, y), (IMG_W, y), (60, 60, 60), 1)

    elif choice == 1:  # деревянный поддон — тёплые коричневые оттенки
        base_c = (int(rng.integers(60, 90)), int(rng.integers(40, 70)), int(rng.integers(20, 50)))
        img[:] = base_c
        for i in range(0, IMG_W, rng.integers(20, 40)):
            shade = int(rng.integers(-15, 15))
            c = tuple(max(0, min(255, base_c[ch] + shade)) for ch in range(3))
            cv2.line(img, (i, 0), (i, IMG_H), c, rng.integers(2, 8))
        noise = rng.integers(0, 12, (IMG_H, IMG_W, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

    else:  # зелёная подстилка (листья) — трава
        base = rng.integers(30, 80, (IMG_H, IMG_W, 3), dtype=np.uint8)
        base[:, :, 1] = np.clip(base[:, :, 1].astype(int) + 40, 0, 255).astype(np.uint8)
        noise_small = rng.integers(0, 25, (IMG_H // 6, IMG_W // 6, 3), dtype=np.uint8)
        noise_up = cv2.resize(noise_small, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        img = cv2.add(base, noise_up)

    return img


def place_objects(img, rng):
    """Размещает 1–4 плода, возвращает список (cls_id, bbox_yolo)."""
    n = int(rng.integers(1, 5))
    annotations = []
    placed = []  # список (x1,y1,x2,y2) для проверки перекрытий

    attempts = 0
    while len(placed) < n and attempts < 50:
        attempts += 1
        cls_id = int(rng.integers(0, NUM_CLASSES))
        rx = int(rng.integers(30, 90))
        ry = int(rng.integers(25, 80))
        cx = int(rng.integers(rx + 5, IMG_W - rx - 5))
        cy = int(rng.integers(ry + 5, IMG_H - ry - 5))

        x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry

        # Проверка пересечения с уже размещёнными
        overlap = False
        for (ox1, oy1, ox2, oy2) in placed:
            iou_x = max(0, min(x2, ox2) - max(x1, ox1))
            iou_y = max(0, min(y2, oy2) - max(y1, oy1))
            if iou_x * iou_y > 0.3 * (2 * rx) * (2 * ry):
                overlap = True
                break

        if overlap:
            continue

        draw_fruit(img, cx, cy, rx, ry, cls_id, rng)
        placed.append((x1, y1, x2, y2))

        # YOLO-формат: нормированные cx, cy, w, h
        norm_cx = (x1 + x2) / 2 / IMG_W
        norm_cy = (y1 + y2) / 2 / IMG_H
        norm_w  = (x2 - x1) / IMG_W
        norm_h  = (y2 - y1) / IMG_H
        annotations.append((cls_id, norm_cx, norm_cy, norm_w, norm_h))

    return annotations


def generate_dataset(output_dir: str, n_train: int = 80, n_val: int = 20):
    rng = np.random.default_rng(SEED)
    base = Path(output_dir)

    splits = {"train": n_train, "val": n_val}
    stats = {cls: 0 for cls in CLASSES}

    for split, count in splits.items():
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(count):
            img = generate_background(rng)
            annots = place_objects(img, rng)

            # Лёгкое гауссово размытие для реализма
            img = cv2.GaussianBlur(img, (3, 3), 0)

            fname = f"{split}_{idx:04d}"
            cv2.imwrite(str(img_dir / f"{fname}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

            with open(lbl_dir / f"{fname}.txt", "w") as f:
                for cls_id, cx, cy, w, h in annots:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    stats[CLASSES[cls_id]] += 1

    # data.yaml — конфигурация датасета для YOLOv8
    yaml_content = f"""# YOLOv8 dataset config — Maturity Classification
path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val

nc: {NUM_CLASSES}
names: {CLASSES}

# Описание классов:
# 0 unripe    — незрелый продукт
# 1 semi_ripe — полузрелый
# 2 ripe      — зрелый / готовый к уборке
# 3 overripe  — перезрелый
"""
    with open(base / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    # Сохраняем статистику
    with open(base / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"train": n_train, "val": n_val, "class_counts": stats}, f, ensure_ascii=False, indent=2)

    print(f"\nДатасет создан: {os.path.abspath(output_dir)}")
    print(f"  Train: {n_train} изображений")
    print(f"  Val:   {n_val} изображений")
    print(f"  Распределение объектов по классам:")
    for cls, cnt in stats.items():
        print(f"    {cls:12s}: {cnt}")
    print(f"  data.yaml: {base / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор синтетического датасета зрелости")
    parser.add_argument("--output", default="dataset", help="Папка для датасета")
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--val",   type=int, default=20)
    args = parser.parse_args()
    generate_dataset(args.output, args.train, args.val)
