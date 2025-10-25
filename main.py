import os
import json
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from scipy.ndimage import label, binary_erosion

# ---------------- CONFIG ----------------
PROTOTYPE_DIR = "/home/btbts22026/adwika/inventory_control/paper/prototype_img"
SCENE_DIR = "/home/btbts22026/adwika/inventory_control/paper/scene_img"
VIS_DIR = "paper_main_visualized_corrected"
OUT_JSON = "paper_main_results_corrected.json"
os.makedirs(VIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Ensemble YOLO models
MODELS = [
    YOLO("/home/btbts22026/adwika/inventory_control/yolov8s.pt"),        # detection
    YOLO("/home/btbts22026/adwika/inventory_control/yolov8s-seg.pt"),    # segmentation
]
for model in MODELS:
    model.to(DEVICE)
    model.fuse()

# Local DINOv2 path
DINOV2_LOCAL = "/home/btbts22026/adwika/inventory_control/dinov2-base"

# ---------------- LOAD DINOv2 ----------------
print("Loading DINOv2...")
processor = AutoImageProcessor.from_pretrained(DINOV2_LOCAL)
dino_model = AutoModel.from_pretrained(DINOV2_LOCAL).to(DEVICE).eval()

# ---------------- HELPER FUNCTIONS ----------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def dino_embedding_from_pil(img_pil, processor, model, device):
    inputs = processor(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    feat = out.last_hidden_state.mean(dim=1).squeeze(0)
    return l2_normalize(feat.detach().cpu().numpy())

def ensemble_predict(models, image_path, conf=0.15, iou=0.20):
    all_boxes = []
    seg_masks = None

    for idx, model in enumerate(models):
        results = model.predict(image_path, conf=conf, iou=iou, device=DEVICE)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') and result.boxes is not None else np.zeros((0, 4))
        all_boxes.append(boxes)

        if idx == 1 and hasattr(result, 'masks') and result.masks is not None:
            seg_masks = result.masks.data.cpu().numpy()

    if all_boxes:
        concat_boxes = np.vstack([b for b in all_boxes if len(b) > 0]) if any(len(b) > 0 for b in all_boxes) else np.zeros((0,4))
    else:
        concat_boxes = np.zeros((0,4))

    if seg_masks is None:
        seg_masks = np.array([])

    return concat_boxes, seg_masks

def filter_masks(masks, min_area_ratio=0.05, img_area=None):
    """Filter small/false positive masks (e.g., under-stack noise) using connected components."""
    if masks.size == 0:
        return masks
    filtered_masks = []
    for mask in masks:
        if mask.size > 0:
            # Erode to remove thin edges/noise
            eroded = binary_erosion(mask > 0.5, structure=np.ones((3,3)))
            labeled, num_features = label(eroded)
            # Keep only large components (filter small blobs)
            large_mask = np.zeros_like(mask)
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if np.sum(component) > (img_area * min_area_ratio) or img_area is None:
                    large_mask[component] = 1
            filtered_masks.append(large_mask.astype(np.float32))
        else:
            filtered_masks.append(mask)
    return np.array(filtered_masks)

# ---------------- LOAD PROTOTYPES ----------------
prototype_feats = {}
box_prototypes = []
cup_prototypes = []
for fname in os.listdir(PROTOTYPE_DIR):
    if fname.lower().endswith((".jpg", ".png")):
        path = os.path.join(PROTOTYPE_DIR, fname)
        img = Image.open(path).convert("RGB")
        feat = dino_embedding_from_pil(img, processor, dino_model, DEVICE)
        prototype_feats[fname] = feat
        if "box" in fname.lower():
            box_prototypes.append(feat)
        elif "cup" in fname.lower():
            cup_prototypes.append(feat)

box_class_feat = l2_normalize(np.mean(box_prototypes, axis=0)) if box_prototypes else None
cup_class_feat = l2_normalize(np.mean(cup_prototypes, axis=0)) if cup_prototypes else None

print("Loaded prototype embeddings:", list(prototype_feats.keys()))
print("Box class feature computed:", box_class_feat is not None)
print("Cup class feature computed:", cup_class_feat is not None)

# ---------------- PROCESS SCENES ----------------
results = {}
for scene_file in sorted(os.listdir(SCENE_DIR)):
    scene_path = os.path.join(SCENE_DIR, scene_file)
    img_bgr = cv2.imread(scene_path)
    if img_bgr is None:
        results[scene_file] = {"matched_class": "None", "count": 0}
        continue

    boxes, masks = ensemble_predict(MODELS, scene_path, conf=0.15, iou=0.20)
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    if masks.size > 0:
        masks = filter_masks(masks, min_area_ratio=0.01, img_area=img_area)  # Filter noise under stacks
    num_masks = masks.shape[0] if masks.size > 0 else 0
    print(f"{scene_file}: Detected {len(boxes)} boxes, {num_masks} filtered masks")

    box_count, cup_count = 0, 0
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    matched_indices = []

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if x2 - x1 < 12 or y2 - y1 < 12:  # Even smaller for partials
            continue
        crop_rgb = img_rgb[y1:y2, x1:x2]
        if crop_rgb.size == 0:
            continue
        crop_pil = Image.fromarray(crop_rgb)
        crop_feat = dino_embedding_from_pil(crop_pil, processor, dino_model, DEVICE)

        box_sim = cosine_similarity_np(crop_feat, box_class_feat) if box_class_feat is not None else -1
        cup_sim = cosine_similarity_np(crop_feat, cup_class_feat) if cup_class_feat is not None else -1

        if box_sim > cup_sim and box_sim > 0.35:
            box_count += 1
            matched_indices.append((idx, 'box'))
        elif cup_sim > box_sim and cup_sim > 0.35:
            cup_count += 1
            matched_indices.append((idx, 'cup'))

    if box_count > cup_count and box_count > 0:
        matched_class = "Box"
    elif cup_count > box_count and cup_count > 0:
        matched_class = "Cup"
    else:
        matched_class = "None"

    total_count = box_count + cup_count
    results[scene_file] = {"matched_class": matched_class, "count": int(total_count)}

    # ---------------- VISUALIZATION ----------------
    vis = img_bgr.copy()

    # Overlay filtered masks with conservative blending (no spill under stacks)
    if masks.size > 0:
        original_h, original_w = img_bgr.shape[:2]
        for i in range(masks.shape[0]):
            mask = masks[i]
            if np.sum(mask) > 0:  # Only if significant area
                resized_mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST) > 0.5
                # Localized overlay: only blend on mask pixels
                mask_overlay = vis.copy()
                green_pixels = np.zeros((original_h, original_w, 3), dtype=np.uint8)
                green_pixels[resized_mask] = (0, 255, 0)
                mask_overlay[resized_mask] = cv2.addWeighted(mask_overlay[resized_mask], 0.6, green_pixels[resized_mask], 0.4, 0)
                vis = cv2.addWeighted(vis, 0.9, mask_overlay, 0.1, 0)  # Very light final blend

    # Class-specific outlines for matched
    box_color = (255, 0, 0)  # Blue for boxes
    cup_color = (0, 255, 0)  # Green for cups
    for idx, cls in matched_indices:
        if len(boxes) > idx:
            x1, y1, x2, y2 = map(int, boxes[idx])
            color = box_color if cls == 'box' else cup_color
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Translucent fill for matched boxes (ensures coverage for partials)
    for idx, cls in matched_indices:
        if len(boxes) > idx:
            x1, y1, x2, y2 = map(int, boxes[idx])
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            vis = cv2.addWeighted(vis, 0.85, overlay, 0.15, 0)  # Lighter for less occlusion

    cv2.imwrite(os.path.join(VIS_DIR, scene_file), vis)
    print(scene_file, "->", matched_class, total_count, f"(boxes: {box_count}, cups: {cup_count})")

# ---------------- SAVE RESULTS ----------------
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
print("Done. JSON saved to", OUT_JSON)


