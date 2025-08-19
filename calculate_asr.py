import math
import cv2
import numpy as np
import onnx
import torch
import os
from onnx2torch import convert
from typing import Optional, List, Tuple, Dict, Any


# ─────────── Вспомогательные функции ───────────
def load_class_names(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]


def preprocess(img: np.ndarray, size: Tuple[int, int], mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    h, w = size
    blob = cv2.resize(img, (w, h)).astype(np.float32)
    return ((blob - mean) / scale).transpose(2, 0, 1)[None, ...]


def make_grid_and_strides(in_h: int, in_w: int, strides: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    centers, stride_map = [], []
    for s in strides:
        fh = math.ceil(in_h / s);
        fw = math.ceil(in_w / s)
        yv, xv = np.meshgrid(np.arange(fh), np.arange(fw), indexing='ij')
        cx = (xv + 0.5) * s;
        cy = (yv + 0.5) * s
        pts = np.stack([cx, cy], -1).reshape(-1, 2)
        centers.append(pts)
        stride_map.append(np.full((pts.shape[0],), s, dtype=np.float32))
    return np.concatenate(centers, 0), np.concatenate(stride_map, 0)


def softmax(x: np.ndarray, axis: int = 2) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0];
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1);
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def postprocess(
        pred: np.ndarray,
        orig_sz: Tuple[int, int],
        in_sz: Tuple[int, int],
        strides: List[int],
        conf_thr: float,
        num_classes: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    orig_h, orig_w = orig_sz
    in_h, in_w = in_sz
    cls_logits = pred[:, :num_classes]
    regs = pred[:, num_classes:]
    N, _ = cls_logits.shape

    centers, stride_map = make_grid_and_strides(in_h, in_w, strides)

    scores_all = 1 / (1 + np.exp(-cls_logits))
    class_ids = np.argmax(scores_all, axis=1)
    scores = scores_all[np.arange(N), class_ids]

    mask = scores > conf_thr
    scores = scores[mask]
    class_ids = class_ids[mask]
    regs = regs[mask]
    centers = centers[mask]
    stride_map = stride_map[mask]

    if scores.size == 0:
        return np.zeros((0, 4)), np.array([]), np.array([]), np.zeros((0, num_classes))

    num_bins = 8
    regs = regs.reshape(-1, 4, num_bins)
    probs = softmax(regs, axis=2)
    bins = np.arange(num_bins, dtype=np.float32)
    dist = (probs * bins).sum(axis=2) * stride_map[:, None]
    l, t, r, b = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
    cx, cy = centers[:, 0], centers[:, 1]
    x1, y1 = cx - l, cy - t
    x2, y2 = cx + r, cy + b
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    sx, sy = orig_w / in_w, orig_h / in_h
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy

    keep = nms(boxes, scores)
    return boxes[keep], scores[keep], class_ids[keep], scores_all[mask][keep]


def draw(
        img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        ids: np.ndarray,
        names: List[str]
) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), sc, cid in zip(boxes, scores, ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{names[cid]} {sc:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def load_model(
        model_path: str
) -> Tuple[torch.nn.Module, int, int, int]:
    onnx_model = onnx.load(model_path)

    # Получаем размеры входа
    input_shape = onnx_model.graph.input[0].type.tensor_type.shape
    H = input_shape.dim[2].dim_value
    W = input_shape.dim[3].dim_value

    # Получаем размерность выхода
    output_shape = onnx_model.graph.output[0].type.tensor_type.shape
    D = output_shape.dim[2].dim_value
    num_classes = D - 32  # 32 - количество смещений (offsets)

    model = convert(onnx_model)
    return model, H, W, num_classes


# ─────────── Основная логика ───────────
def run_experiment(
        model_path: str = "nanodet.onnx",
        image_dir: str = "dataset",
        classes_path: Optional[str] = None,
        conf_threshold: float = 0.3,
        patch_size: float = 0.4,
        patch_name: str = "dpatch5000",
        results_dir: Optional[str] = None
) -> Dict[str, Any]:
    # Вычисляем производные пути
    patch_path = f"{patch_name}.png"
    if results_dir is None:
        results_dir = f"patched_{patch_name}"

    # Загрузка модели
    model, H, W, num_classes = load_model(model_path)

    # Загрузка имен классов
    class_names = load_class_names(classes_path)
    if not class_names:
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Создание директории для результатов
    os.makedirs(results_dir, exist_ok=True)

    # Константы для предобработки
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    scale = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    strides = [8, 16, 32, 64]

    # Статистика
    total_class0_objects = 0
    successful_attacks = 0
    confidence_drops = []

    # Обработка изображений
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        # Загрузка изображения
        orig = cv2.imread(image_file)
        if orig is None:
            print(f"Не удалось загрузить {image_file}")
            continue
        orig_h, orig_w = orig.shape[:2]

        # Детекция на оригинальном изображении
        blob = preprocess(orig, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes, scores, cls_ids, scores_all = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

        # Наложение патча
        patched = orig.copy()
        class0_indices = np.where(cls_ids == 0)[0]
        total_class0_objects += len(class0_indices)

        for i in class0_indices:
            x1, y1, x2, y2 = map(int, boxes[i])
            width = int(x2 - x1)
            height = int(y2 - y1)

            patch = cv2.imread(patch_path, cv2.IMREAD_UNCHANGED)
            if patch is None:
                print(f"Не удалось загрузить патч {patch_path}")
                continue

 #           patch_resized = cv2.resize(
 #               patch,
 #               (int(width * patch_size), int(height * patch_size))
 #           )
            patch_resized = patch
            try:
#                patched[y1:y1 + patch_resized.shape[0],
#                x1:x1 + patch_resized.shape[1]] = patch_resized
                patched[:patch_resized.shape[0], :patch_resized.shape[1]] = patch_resized
            except Exception as e:
                print(f"Ошибка при наложении патча: {e}")
                continue

        # Детекция на патченом изображении
        blob = preprocess(patched, (H, W), mean, scale)
        pred = model(torch.from_numpy(blob))[0].detach().numpy()
        boxes_p, scores_p, cls_ids_p, scores_all_p = postprocess(
            pred, (orig_h, orig_w), (H, W), strides, conf_threshold, num_classes
        )

        # Расчет метрик атаки
        for idx in class0_indices:
            orig_box = boxes[idx]
            orig_score = scores_all[idx, 0]
            found = False

            for j, patched_box in enumerate(boxes_p):
                if cls_ids_p[j] == 0:
                    iou = calculate_iou(orig_box, patched_box)
                    if iou > 0.5:
                        found = True
                        patched_score = scores_all_p[j, 0]
                        confidence_drops.append(orig_score - patched_score)
                        break

            if not found:
                successful_attacks += 1
                confidence_drops.append(orig_score)

        # Визуализация результатов
        vis_clean = draw(orig.copy(), boxes, scores, cls_ids, class_names)
        vis_patched = draw(patched.copy(), boxes_p, scores_p, cls_ids_p, class_names)
        result_img = np.hstack([vis_clean, vis_patched])

        result_path = os.path.join(
            results_dir,
            f"result_{os.path.basename(image_file)}"
        )
        cv2.imwrite(result_path, result_img)
        print(f"Результат сохранен: {result_path}")

    # Расчет итоговых метрик
    metrics = {
        'total_class0_objects': total_class0_objects,
        'successful_attacks': successful_attacks,
        'confidence_drops': confidence_drops
    }

    if total_class0_objects > 0:
        metrics['asr'] = successful_attacks / total_class0_objects
        metrics['mean_confidence_drop'] = np.mean(confidence_drops)
    else:
        metrics['asr'] = 0.0
        metrics['mean_confidence_drop'] = 0.0

    # Вывод результатов
    print(f"\nНазвание атаки: {patch_name}")
    print(f"Объекты класса 0: {total_class0_objects}")
    print(f"Успешные атаки: {successful_attacks}")

    if total_class0_objects > 0:
        print(f"ASR: {metrics['asr']:.4f}")
        print(f"Среднее падение уверенности: {metrics['mean_confidence_drop']:.4f}")
    else:
        print("Объекты класса 0 не обнаружены")

    return metrics


# ─────────── Точка входа ───────────
if __name__ == "__main__":
    # Вызов с параметрами по умолчанию
    results = run_experiment(patch_name='selfmade_patch')

    # Пример вызова с кастомными параметрами:
    # run_experiment(
    #     model_path="custom_model.onnx",
    #     image_dir="custom_dataset",
    #     conf_threshold=0.25,
    #     patch_size=0.35,
    #     patch_name="custom_patch",
    #     results_dir="custom_results"
    # )