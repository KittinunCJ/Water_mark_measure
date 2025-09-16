# backend/infer_service.py
# -*- coding: utf-8 -*-
"""
ตัวห่อโมเดล:
- โหลด YOLO + Fusion model ครั้งเดียว แล้ว reuse
- ถ้า weights/ckpt/scaler ไม่มีในดิสก์และมี *_URL ใน ENV -> auto-download มาเก็บที่ backend/weights/
- รับ path รูป -> คืนค่าความสูงรอยน้ำ (เมตร, ปัด 2 ตำแหน่ง)
"""
from pathlib import Path
import os, json, hashlib
import numpy as np
import pandas as pd
import torch
import httpx
from ultralytics import YOLO
from dotenv import load_dotenv

# โหลด .env ได้ทั้ง root และ backend
HERE = Path(__file__).resolve().parent                  # .../backend
ROOT = HERE.parent
load_dotenv(ROOT / ".env")
load_dotenv(HERE / ".env")

# ใช้โค้ดเดิมของคุณ
from .infer_res34_adjusted import (
    WATER_CLASS, REF_CLASSES,
    combine_water_masks, clean_mask, waterline_y_curve, ref_sizes_from_result,
    FusionRegressor, load_checkpoint, add_engineered_features, apply_scaler_df,
    make_feature_matrix, load_image_tensor, r2
)

# ====== Path/Env ======
WEIGHTS_DIR       = HERE / "weights"; WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_YOLO_PATH = Path(os.environ.get("WEIGHTS_YOLO", str(WEIGHTS_DIR / "best.pt")))
CKPT_FUSION_PATH  = Path(os.environ.get("CKPT_FUSION",  str(WEIGHTS_DIR / "fusion_resnet34_best.pt")))
SCALER_JSON_PATH  = Path(os.environ.get("SCALER_JSON",  str(WEIGHTS_DIR / "scaler.json")))

# Optional URLs (ใช้เมื่อไฟล์หาย)
WEIGHTS_YOLO_URL  = os.environ.get("WEIGHTS_YOLO_URL", "")
CKPT_FUSION_URL   = os.environ.get("CKPT_FUSION_URL", "")
SCALER_JSON_URL   = os.environ.get("SCALER_JSON_URL", "")

# Optional SHA256 for integrity check
WEIGHTS_YOLO_SHA256 = os.environ.get("WEIGHTS_YOLO_SHA256", "")
CKPT_FUSION_SHA256  = os.environ.get("CKPT_FUSION_SHA256", "")
SCALER_JSON_SHA256  = os.environ.get("SCALER_JSON_SHA256", "")

IMG_SIZE_YOLO  = int(os.environ.get("IMG_SIZE_YOLO", "960"))
CONF_YOLO      = float(os.environ.get("CONF_YOLO", "0.35"))
IOU_YOLO       = float(os.environ.get("IOU_YOLO",  "0.50"))

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Utils: download if missing ======
def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_download(url: str, dst: Path, sha256: str = "") -> None:
    """
    ถ้า url ถูกกำหนดและไฟล์ปลายทางยังไม่มี -> ดาวน์โหลดด้วย httpx แบบ stream
    ถ้ามี sha256 -> ตรวจสอบก่อนใช้งาน
    รองรับ GitHub Releases (public). ถ้ามี GITHUB_TOKEN จะใส่ header ให้
    """
    if not url:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    headers = {}
    token = os.environ.get("GITHUB_TOKEN", "")
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/octet-stream"

    with httpx.stream("GET", url, headers=headers, timeout=None, follow_redirects=True) as r:
        r.raise_for_status()
        tmp = dst.with_suffix(dst.suffix + ".part")
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                if chunk:
                    f.write(chunk)
        if sha256:
            got = _sha256_of(tmp)
            if got.lower() != sha256.lower():
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"SHA256 mismatch for {dst.name}: expected {sha256}, got {got}")
        tmp.rename(dst)

def _ensure_local_files():
    # YOLO weights
    if not WEIGHTS_YOLO_PATH.exists() and WEIGHTS_YOLO_URL:
        _maybe_download(WEIGHTS_YOLO_URL, WEIGHTS_YOLO_PATH, WEIGHTS_YOLO_SHA256)
    # Fusion ckpt
    if not CKPT_FUSION_PATH.exists() and CKPT_FUSION_URL:
        _maybe_download(CKPT_FUSION_URL, CKPT_FUSION_PATH, CKPT_FUSION_SHA256)
    # scaler.json
    if not SCALER_JSON_PATH.exists() and SCALER_JSON_URL:
        _maybe_download(SCALER_JSON_URL, SCALER_JSON_PATH, SCALER_JSON_SHA256)

# ====== Lazy singletons ======
_yolo            = None
_fusion_model    = None
_feat_names      = None
_img_size_fusion = 384
_scaler          = None

def _lazy_init():
    """โหลดโมเดล/สเกลเลอร์ครั้งเดียว (ดาวน์โหลดถ้าหาย)"""
    global _yolo, _fusion_model, _feat_names, _img_size_fusion, _scaler

    _ensure_local_files()

    if _yolo is None:
        if not WEIGHTS_YOLO_PATH.exists():
            raise FileNotFoundError(f"YOLO weights not found: {WEIGHTS_YOLO_PATH}")
        _yolo = YOLO(str(WEIGHTS_YOLO_PATH))
        _yolo.to(str(DEVICE))

    if _fusion_model is None:
        if not CKPT_FUSION_PATH.exists():
            raise FileNotFoundError(f"Fusion checkpoint not found: {CKPT_FUSION_PATH}")
        _fusion_model, _feat_names, _img_size_fusion = load_checkpoint(
            str(CKPT_FUSION_PATH), default_img_size=384
        )
        _fusion_model.to(DEVICE).eval()

    if _scaler is None and SCALER_JSON_PATH.exists():
        try:
            _scaler = json.load(open(SCALER_JSON_PATH, "r", encoding="utf-8"))
        except Exception:
            _scaler = None  # ไม่มี scaler ก็ใช้ได้

# ====== Public API ======
def predict_height_m(image_path: str) -> float:
    """
    รับ path รูป -> คืนความสูงรอยน้ำหน่วย 'เมตร' (ปัด 2 ตำแหน่ง)
    - ถ้าตรวจไม่เจอ waterline จะคืน NaN
    """
    try:
        _lazy_init()

        # YOLO
        r = _yolo.predict(
            source=image_path,
            imgsz=IMG_SIZE_YOLO,
            conf=CONF_YOLO,
            iou=IOU_YOLO,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]
        H, W = r.orig_img.shape[:2]

        # water mask -> waterline
        wm = combine_water_masks(r, WATER_CLASS)
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)
        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        if y_curve is None:
            return float("nan")

        pixel_height = (H - 1) - int(y_med)
        ref_dict = ref_sizes_from_result(r)

        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        X, _ = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)
        im = load_image_tensor(image_path, _img_size_fusion).to(DEVICE)

        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            y = _fusion_model(im, X_t).detach().cpu().item()

        return float(r2([y], 2)[0])
    except Exception:
        # อย่าทำให้ทั้ง API ล่ม—คืน NaN แล้วไป calibrate ต่อฝั่ง app
        return float("nan")
