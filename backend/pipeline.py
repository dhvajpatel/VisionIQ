"""
pipeline.py — VisionIQ inference core (desktop edition, fully offline).

All model weights are bundled inside the installer (no internet required).
Paths are resolved automatically for both dev and PyInstaller environments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger("visioniq.pipeline")

# ── Resolve model cache path ──────────────────────────────────────────────────
# In PyInstaller bundle: sys._MEIPASS/_internal points to extracted files
# The models_cache/ folder is bundled alongside the exe
def _get_models_cache() -> Path:
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle — models_cache is next to the exe
        exe_dir = Path(sys.executable).parent
        candidate = exe_dir / "models_cache"
        if candidate.exists():
            return candidate
        # Also check _internal (onedir layout)
        candidate2 = Path(sys._MEIPASS).parent / "models_cache"
        if candidate2.exists():
            return candidate2
        # Fallback: same dir as exe
        return exe_dir / "models_cache"
    else:
        # Dev mode — models_cache is in backend/
        return Path(__file__).parent / "models_cache"

MODELS_CACHE = _get_models_cache()
HF_CACHE     = MODELS_CACHE / "hub"
TORCH_CACHE  = MODELS_CACHE / "torch"

# Force all HuggingFace and torch downloads to the bundled cache
os.environ["HF_HOME"]               = str(MODELS_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"]    = str(HF_CACHE)
os.environ["TORCH_HOME"]            = str(TORCH_CACHE)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"   # ← KEY: never try to download

logger.info("Models cache: %s (exists=%s)", MODELS_CACHE, MODELS_CACHE.exists())

# ── constants ────────────────────────────────────────────────────────────────

# ImageNet-1k: indices 0–397 are animal synsets (fish → mammals).
ANIMAL_CLASS_MAX_INDEX = 397

# MTCNN minimum detection confidence.
# Lower threshold = more faces detected (catches partially occluded/tilted faces)
FACE_CONFIDENCE_THRESHOLD = 0.75

# Cap faces per image — raised to handle large group photos
MAX_FACES = 20

# Crop padding fractions — more padding gives the model more facial context
FACE_PAD_X     = 0.25   # 25% each side — captures ears and hair
FACE_PAD_Y_TOP = 0.20   # 20% above — captures forehead/hair
FACE_PAD_Y_BOT = 0.10   # 10% below — captures chin

# Context crop: extend below chin (fraction of face height)
# Reduced — clothing adds noise for gender classification
BODY_CONTEXT_FACTOR = 0.25

# Temperature sharpening — raised closer to 1.0 to avoid over-confident wrong predictions
# T=0.65 was too aggressive: a 60% confidence would become 95%+ causing wrong labels
# T=0.85 is gentler: only strongly confident predictions get amplified
SHARPEN_TEMPERATURE = 0.85

# Minimum crop dimension in pixels
MIN_CROP_SIZE = 32

# Normalise varied label strings → "Male" / "Female"
GENDER_LABEL_MAP = {
    "male":   "Male",  "man":    "Male",  "boy":    "Male",
    "female": "Female","woman":  "Female","girl":   "Female",
    "0":      "Male",  "1":      "Female",
}

# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    box: List[int]
    face_confidence: float
    gender: str
    gender_confidence: float


@dataclass
class PredictionResult:
    category: str                       # "person" | "animal" | "unknown"
    faces: List[FaceResult] = field(default_factory=list)
    animal_label: Optional[str] = None
    animal_confidence: Optional[float] = None
    fallback_label: Optional[str] = None
    fallback_confidence: Optional[float] = None


# ── engine ───────────────────────────────────────────────────────────────────

class VisionIQEngine:
    """
    Loads all models in float16 to stay within Render's 512 MB RAM limit.
    """

    def __init__(self):
        self.device = torch.device("cpu")  # Render has no GPU
        logger.info("Device: cpu (float16 mode)")

        # ── MTCNN face detector (~50 MB, float32 — MTCNN doesn't support f16) ──
        logger.info("Loading MTCNN…")
        self.face_detector = MTCNN(
            keep_all=True,
            device=self.device,
            post_process=False,
            min_face_size=20,
        )

        # ── ViT-B/16 gender classifier loaded in float16 (~165 MB) ──────────
        logger.info("Loading ViT-B/16 in float16… (cache: %s)", HF_CACHE)
        MODEL_ID = "rizvandwiki/gender-classification"
        self.gender_processor = AutoImageProcessor.from_pretrained(
            MODEL_ID,
            cache_dir=str(HF_CACHE),
            local_files_only=True,
        )
        self.gender_model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            cache_dir=str(HF_CACHE),
            local_files_only=True,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()
        # Build id→label map
        self.gender_id2label = self.gender_model.config.id2label

        # ── ResNet50 in float16 (~50 MB) ──────────────────────────────────────
        logger.info("Loading ResNet50 in float16…")
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.animal_classifier = resnet50(weights=weights).half().eval()
        self.animal_transform  = weights.transforms()
        self.animal_categories = weights.meta["categories"]

        self._loaded = True
        logger.info("All models ready.")

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _norm_label(raw: str) -> str:
        return GENDER_LABEL_MAP.get(raw.strip().lower(), raw.title())

    def _gender_scores(self, crop: Image.Image) -> dict:
        """Run ViT-B/16 (float16) on crop → normalised {'Male': p, 'Female': p}."""
        # Upscale small crops — ViT works better with larger inputs
        # The model was trained on 224x224; small faces lose detail at that resolution
        if crop.width < 112 or crop.height < 112:
            scale = max(112 / crop.width, 112 / crop.height)
            new_w = max(int(crop.width  * scale), 112)
            new_h = max(int(crop.height * scale), 112)
            crop  = crop.resize((new_w, new_h), Image.LANCZOS)

        inputs = self.gender_processor(images=crop, return_tensors="pt")
        # Cast inputs to float16 to match model weights
        inputs = {k: v.half() if v.dtype == torch.float32 else v
                  for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.gender_model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze(0)

        scores: dict = {}
        for idx, prob in enumerate(probs):
            raw   = self.gender_id2label.get(idx, str(idx))
            label = self._norm_label(raw)
            scores[label] = scores.get(label, 0.0) + float(prob)
        scores.setdefault("Male",   0.0)
        scores.setdefault("Female", 0.0)
        total = scores["Male"] + scores["Female"]
        if total > 1e-6:
            scores = {k: v / total for k, v in scores.items()}
        return scores

    @staticmethod
    def _sharpen(male_prob: float, female_prob: float) -> Tuple[float, float]:
        """Temperature sharpening — pushes scores away from 0.5."""
        p_m = max(male_prob,   1e-6)
        p_f = max(female_prob, 1e-6)
        logit_m = torch.tensor(p_m).log() / SHARPEN_TEMPERATURE
        logit_f = torch.tensor(p_f).log() / SHARPEN_TEMPERATURE
        probs   = F.softmax(torch.stack([logit_m, logit_f]), dim=0)
        return float(probs[0]), float(probs[1])

    def _classify_gender(self, face_crop: Image.Image, ctx_crop: Image.Image) -> Tuple[str, float]:
        """
        Run ViT-B/16 on face crop (primary) and context crop (secondary).
        Weight face crop more heavily — it has cleaner facial features.
        Only use context crop as a tiebreaker for near-50/50 cases.
        """
        sf = self._gender_scores(face_crop)
        sc = self._gender_scores(ctx_crop)

        # Weight face crop 70%, context crop 30%
        # (context crop was causing errors by including clothing from other people)
        avg_m = sf["Male"]   * 0.70 + sc["Male"]   * 0.30
        avg_f = sf["Female"] * 0.70 + sc["Female"] * 0.30
        total = avg_m + avg_f
        if total > 1e-6:
            avg_m /= total
            avg_f /= total

        # Only sharpen if there is clear confidence (>= 55%)
        # — avoids flipping uncertain predictions to wrong label at high confidence
        if max(avg_m, avg_f) >= 0.55:
            sharp_m, sharp_f = self._sharpen(avg_m, avg_f)
        else:
            sharp_m, sharp_f = avg_m, avg_f

        label = "Female" if sharp_f >= sharp_m else "Male"
        return label, round(sharp_f if label == "Female" else sharp_m, 4)

    @staticmethod
    def _tight_face_crop(image: Image.Image, box: List[int]) -> Image.Image:
        x0, y0, x1, y1 = box
        w, h   = x1 - x0, y1 - y0
        pad_x  = int(w * FACE_PAD_X)
        pad_t  = int(h * FACE_PAD_Y_TOP)
        pad_b  = int(h * FACE_PAD_Y_BOT)
        crop = image.crop((
            max(0, x0 - pad_x), max(0, y0 - pad_t),
            min(image.width,  x1 + pad_x),
            min(image.height, y1 + pad_b),
        ))
        if min(crop.width, crop.height) < MIN_CROP_SIZE:
            scale = MIN_CROP_SIZE / min(crop.width, crop.height)
            crop = crop.resize(
                (max(int(crop.width * scale), MIN_CROP_SIZE),
                 max(int(crop.height * scale), MIN_CROP_SIZE)),
                Image.LANCZOS,
            )
        return crop

    @staticmethod
    def _context_crop(image: Image.Image, box: List[int]) -> Image.Image:
        x0, y0, x1, y1 = box
        face_h = y1 - y0
        pad_w  = int((x1 - x0) * 0.20)
        crop = image.crop((
            max(0, x0 - pad_w),
            y0,
            min(image.width,  x1 + pad_w),
            min(image.height, y1 + int(face_h * BODY_CONTEXT_FACTOR)),
        ))
        if min(crop.width, crop.height) < MIN_CROP_SIZE:
            scale = MIN_CROP_SIZE / min(crop.width, crop.height)
            crop = crop.resize(
                (max(int(crop.width * scale), MIN_CROP_SIZE),
                 max(int(crop.height * scale), MIN_CROP_SIZE)),
                Image.LANCZOS,
            )
        return crop

    def _run_animal_classifier(self, image: Image.Image) -> Tuple[int, str, float]:
        # animal_transform returns float32; cast to float16 to match model
        tensor = self.animal_transform(image).unsqueeze(0).half()
        with torch.no_grad():
            probs = F.softmax(self.animal_classifier(tensor), dim=1).squeeze(0)
        top_idx = int(torch.argmax(probs).item())
        return top_idx, self.animal_categories[top_idx], round(float(probs[top_idx]), 4)

    # ── main predict ─────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> PredictionResult:
        image = image.convert("RGB")
        boxes, probs = self.face_detector.detect(image)

        faces: List[FaceResult] = []
        if boxes is not None:
            # Keep only high-confidence detections, sorted best-first, capped
            candidates = sorted(
                [(b, p) for b, p in zip(boxes, probs)
                 if p is not None and p >= FACE_CONFIDENCE_THRESHOLD],
                key=lambda x: x[1], reverse=True
            )[:MAX_FACES]

            for box, prob in candidates:
                x0, y0, x1, y1 = [int(max(v, 0)) for v in box]
                x1 = min(x1, image.width)
                y1 = min(y1, image.height)
                if x1 <= x0 or y1 <= y0:
                    continue
                try:
                    face_crop = self._tight_face_crop(image, [x0, y0, x1, y1])
                    ctx_crop  = self._context_crop(image,   [x0, y0, x1, y1])
                    gender, conf = self._classify_gender(face_crop, ctx_crop)
                    faces.append(FaceResult(
                        box=[x0, y0, x1, y1],
                        face_confidence=round(float(prob), 4),
                        gender=gender,
                        gender_confidence=conf,
                    ))
                except Exception as exc:
                    logger.warning("Skipping face %s: %s", [x0,y0,x1,y1], exc)

        if faces:
            return PredictionResult(category="person", faces=faces)

        top_idx, label, top_prob = self._run_animal_classifier(image)
        if top_idx <= ANIMAL_CLASS_MAX_INDEX:
            return PredictionResult(category="animal", animal_label=label, animal_confidence=top_prob)
        return PredictionResult(category="unknown", fallback_label=label, fallback_confidence=top_prob)


# Backwards-compatible alias (app.py imports DivyaChakshuEngine)
DivyaChakshuEngine = VisionIQEngine
