## YOLOv8-DINOv2 Approach for Vision based Object Counting

**A robust vision-based inventory monitoring pipeline leveraging the strengths of YOLOv8 [1] and DINOv2 [2] for accurate object localization, classification, and counting in cluttered, stacked scenes. Complete network architecture, training parameters, and implementation are provided for reproducibility.**

---


## Architecture

- **YOLOv8 Detection & Segmentation:** [1]
  - Dual models (detection, segmentation) provide bounding boxes and pixel-wise masks. Models are fused for higher recall and robustness.
  - Anchor-free, real-time performance. Ensemble maximizes detection confidence and mask quality.

- **DINOv2 Vision Transformers:** [2]
  - Extract highly semantic, scale-invariant visual features for both prototype images and detected crops.
  - Enables object class assignment with few-shot/few-image reference matching (prototypes).
  - Provides robust embeddings for downstream similarity comparison, even in challenging scenes.

- **Prototype Matching:**
  - Each detected crop is embedded and classified using cosine similarity against precomputed reference vectors.
  - Class-wise counting supports dynamic inventory changes (new objects = new reference images, no retraining required).

- **Postprocessing:**
  - Morphological filtering and area-based suppression remove false positives and spurious small regions (covers stacked/under-stack artifacts).
  - Visualization overlays use color-coded boxes/masks, supporting both qualitative and quantitative evaluation.

- **Outputs:**
  - Results are saved in JSON format per image, with visualized overlays for inspection.

---

## Implementation & Training Parameters

- **Frameworks:** PyTorch, ultralytics-YOLOv8, HuggingFace transformers (DINOv2), OpenCV, NumPy, SciPy, PIL.
- **Detection Model:** YOLOv8 detection/segmentation, fine-tuned for stacked warehouse scenes.
  - Confidence threshold: `0.15`
  - IOU threshold: `0.20`
  - NMS enabled, fused ensemble for maximum recall.
- **Embeddings:** DINOv2-base, self-supervised
  - All reference (“prototype”) images processed once to extract stable class vectors.
  - Scene crop matching threshold: `0.35` cosine similarity (recommended for separation of classes).
- **Mask Filtering:** 
  - Area threshold: Typically `1%` of image area, tunable. 
  - Binary erosion kernel: `3x3` to remove edge noise.
- **Class Assignment:** 
  - Mean class vector for each object group (“Box”, “Cup”) from available references.
  - Each scene is processed for matched class and object count—fully automated, real time (if GPU available).

**Complete details of the network architecture, training parameters, and implementation are available in this repository and can be found in the code, configuration files, and logs. Refer to our published paper for a comprehensive evaluation and ablation studies.**

---

## Installation

# Clone the Repository

```bash
git clone https://github.com/singhadwika/Hybrid-YOLOv8-DINOv2-Inventory.git
cd Hybrid-YOLOv8-DINOv2-Inventory 
```
# Install Dependencies
```bash
pip install -r requirements.txt
```
# Organize Data
Place your images in the following directories:
Prototype images: paper/prototype_img/
Scene images: paper/scene_img/

# Run Main Script
```bash
python main.py
```

---
## References

- [1] Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics  
- [2] DINOv2 Vision Transformer: https://huggingface.co/facebook/dinov2-base

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{singh2025yolov8_dino,
  author    = {Adwika Singh and Bipul Kumar Das and Teena Sharma and Nishchal K. Verma},
  title     = {YOLOv8-DINOv2 Approach for Vision based Object Counting},
  booktitle = {Proceedings of the International Conference on Intelligent Human Computer Interaction},
  publisher = {Springer},
  address   = {Banasthali Vidyapith, Jaipur, Rajasthan, India},
  year      = {2025},
  url       = {https://github.com/singhadwika/Hybrid-YOLOv8-DINOv2-Inventory},
  note      = {Code and implementation available on GitHub}
}

```
