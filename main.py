import os
import json
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from scipy.ndimage import label


# ---------------- CONFIG ----------------
PROTOTYPE_DIR = "/home/btbts22026/adwika/inventory_control/paper/prototype_img"
SCENE_DIR = "/home/btbts22026/adwika/inventory_control/paper/scene_img"
VIS_DIR = "paper_visualized"
OUT_JSON = "paper_results.json"
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


def ensemble_predict(models, image_path, conf=0.25, iou=0.3):
    all_boxes = []
    all_masks = []


    for model in models:
        results = model.predict(image_path, conf=conf, iou=iou, device=DEVICE)
        result = results[0]  # take first result


        # Boxes
        boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result, 'boxes') and result.boxes is not None else np.zeros((0, 4))
        all_boxes.append(boxes)


        # Masks (only for segmentation models)
        masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') and result.masks is not None else np.zeros((0,1,1))
        all_masks.append(masks)


    # Concatenate all boxes
    if all_boxes:
        concat_boxes = np.vstack([b for b in all_boxes if len(b) > 0]) if any(len(b) > 0 for b in all_boxes) else np.zeros((0,4))
    else:
        concat_boxes = np.zeros((0,4))


    # Fuse masks
    if all_masks and any(len(m) > 0 for m in all_masks):
        max_h = max(m.shape[1] for m in all_masks if len(m) > 0)
        max_w = max(m.shape[2] for m in all_masks if len(m) > 0)
        resized_masks = []
        for m in all_masks:
            if len(m) > 0:
                resized = np.array([cv2.resize(mm.astype(np.float32), (max_w, max_h), interpolation=cv2.INTER_NEAREST) for mm in m])
            else:
                resized = np.zeros((0, max_h, max_w), dtype=np.float32)
            resized_masks.append(resized)
        fused_masks = np.max(np.concatenate([rm for rm in resized_masks if len(rm) > 0], axis=0), axis=0)
    else:
        fused_masks = np.array([])


    return concat_boxes, fused_masks


def count_from_masks(masks):
    if masks.size == 0:
        return 0
    combined_mask = masks > 0.5
    labeled, num_features = label(combined_mask)
    return num_features


# ---------------- LOAD PROTOTYPES ----------------
prototype_feats = {}
for fname in os.listdir(PROTOTYPE_DIR):
    if fname.lower().endswith((".jpg", ".png")):
        path = os.path.join(PROTOTYPE_DIR, fname)
        img = Image.open(path).convert("RGB")
        prototype_feats[fname] = dino_embedding_from_pil(img, processor, dino_model, DEVICE)
print("Loaded prototype embeddings:", list(prototype_feats.keys()))


# ---------------- PROCESS SCENES ----------------
results = {}
for scene_file in sorted(os.listdir(SCENE_DIR)):
    scene_path = os.path.join(SCENE_DIR, scene_file)
    img_bgr = cv2.imread(scene_path)
    if img_bgr is None:
        results[scene_file] = {"matched_prototype": "None", "count": 0}
        continue


    # Ensemble Prediction
    boxes, masks = ensemble_predict(MODELS, scene_path)


    # Prototype Matching
    img_pil = Image.open(scene_path).convert("RGB")
    scene_feat = dino_embedding_from_pil(img_pil, processor, dino_model, DEVICE)
    best_proto, best_score = "None", -1
    for proto_name, proto_feat in prototype_feats.items():
        score = cosine_similarity_np(scene_feat, proto_feat)
        if score > best_score:
            best_proto, best_score = proto_name, score


    # Count from fused masks
    count = count_from_masks(masks)


    results[scene_file] = {"matched_prototype": best_proto, "count": int(count)}


    # ---------------- VISUALIZATION ----------------
    vis = img_bgr.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)


    if masks.size > 0:
        original_h, original_w = img_bgr.shape[:2]
        resized_mask = cv2.resize(masks.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST) > 0.5
        mask_overlay = np.zeros_like(vis)
        mask_overlay[resized_mask] = (0, 255, 0)
        vis = cv2.addWeighted(vis, 0.5, mask_overlay, 0.5, 0)


    cv2.imwrite(os.path.join(VIS_DIR, scene_file), vis)
    print(scene_file, "->", best_proto, count)


# ---------------- SAVE RESULTS ----------------
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=4)
print("Done. JSON saved to", OUT_JSON)

understand this code. it has to be optimized as not all objects are being detected and not all are correct highlighted in green

given are our results:

{
    "Box_1.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 0
    },
    "Box_2.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 1
    },
    "Box_3.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 2
    },
    "Box_4.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 2
    },
    "Box_5.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 1
    },
    "Box_6.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 4
    },
    "Box_7.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 3
    },
    "Cup_1.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 0
    },
    "Cup_2.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 2
    },
    "Cup_3.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 1
    },
    "Cup_4.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 5
    },
    "Cup_5.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 5
    }
}



correct outshould be like below:
{
    "Box_1.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 4
    },
    "Box_2.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 4
    },
    "Box_3.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 4
    },
    "Box_4.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 5
    },
    "Box_5.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 5
    },
    "Box_6.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 5
    },
    "Box_7.jpg": {
        "matched_prototype": "Box.jpg",
        "count": 5
    },
    "Cup_1.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 9
    },
    "Cup_2.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 10
    },
    "Cup_3.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 10
    },
    "Cup_4.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 5
    },
    "Cup_5.jpg": {
        "matched_prototype": "Cup.jpg",
        "count": 5
    }
}


###################

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

