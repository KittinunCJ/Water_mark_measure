# backend/infer_service.py
# -*- coding: utf-8 -*-
"""
ตัวห่อโมเดล:
- โหลด YOLO + Fusion model ครั้งเดียว แล้ว reuse
- รับ path รูป -> คืนค่าความสูงรอยน้ำ (เมตร, ปัด 2 ตำแหน่ง)
- รองรับ CKPT จาก URL (auto-download) หรือพาธโลคอล
"""
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import torch
import requests
from ultralytics import YOLO

# ใช้ของเดิมจากไฟล์คุณ
from .infer_res34_adjusted import (
    WATER_CLASS, REF_CLASSES,
    combine_water_masks, clean_mask, waterline_y_curve, ref_sizes_from_result,
    FusionRegressor, load_checkpoint, add_engineered_features, apply_scaler_df,
    make_feature_matrix, load_image_tensor, r2
)

# ====== Path/Env ======
BASE_DIR    = Path(__file__).resolve().parent                  # .../backend
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_YOLO = os.environ.get("WEIGHTS_YOLO", str(WEIGHTS_DIR / "best.pt"))

# รองรับทั้ง CKPT_FUSION และ MODEL_URL (ใช้ CKPT_FUSION ก่อน ถ้าไม่มีค่อยตกไปใช้ MODEL_URL)
CKPT_FUSION = os.environ.get("CKPT_FUSION") or os.environ.get("MODEL_URL") or str(WEIGHTS_DIR / "fusion_resnet34_best.pt")

SCALER_JSON = os.environ.get("SCALER_JSON", str(WEIGHTS_DIR / "scaler.json"))

IMG_SIZE_YOLO = int(os.environ.get("IMG_SIZE_YOLO", "960"))
CONF_YOLO     = float(os.environ.get("CONF_YOLO", "0.35"))
IOU_YOLO      = float(os.environ.get("IOU_YOLO",  "0.50"))

GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN", "").strip()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Lazy singletons ======
_yolo             = None
_fusion_model     = None
_feat_names       = None
_img_size_fusion  = 384
_scaler           = None


# ====== helpers: path/URL handling ======
def _is_url(s: str) -> bool:
    return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

def _download_if_url(url_or_path: str, dst_dir: Path) -> str:
    """
    ถ้าเป็น URL -> ดาวน์โหลดไปไว้ที่ dst_dir แล้วคืนพาธไฟล์
    ถ้าเป็นพาธอยู่แล้ว -> คืนพาธเดิม
    """
    if not url_or_path:
        return url_or_path

    if not _is_url(url_or_path):
        return url_or_path  # local path

    dst_dir.mkdir(parents=True, exist_ok=True)
    fname = url_or_path.split("/")[-1] or "model.ckpt"
    out_path = dst_dir / fname
    if out_path.exists() and out_path.stat().st_size > 0:
        return str(out_path)

    headers = {}
    if GITHUB_TOKEN and "github.com" in url_or_path:
        # รองรับดาวน์โหลดจาก GitHub Releases/Raw private
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
        headers["Accept"] = "application/octet-stream"

    # ดาวน์โหลด
    r = requests.get(url_or_path, headers=headers, timeout=300)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return str(out_path)

def _ensure_exists(p: str | Path, label: str):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return str(p)


def _lazy_init():
    """โหลดโมเดล/สเกลเลอร์ครั้งเดียว (พร้อม auto-download ถ้าให้มาเป็น URL)"""
    global _yolo, _fusion_model, _feat_names, _img_size_fusion, _scaler
    global WEIGHTS_YOLO, CKPT_FUSION, SCALER_JSON  # อัปเดตค่าเป็นไฟล์ที่ดาวน์โหลดแล้ว

    # แปลง URL -> local file (cache)
    WEIGHTS_YOLO = _download_if_url(WEIGHTS_YOLO, WEIGHTS_DIR)
    CKPT_FUSION  = _download_if_url(CKPT_FUSION,  WEIGHTS_DIR)
    SCALER_JSON  = _download_if_url(SCALER_JSON,  WEIGHTS_DIR)  # เผื่อใส่ URL มาด้วย

    if _yolo is None:
        _ensure_exists(WEIGHTS_YOLO, "YOLO weights")
        _yolo = YOLO(WEIGHTS_YOLO)
        # Render ส่วนใหญ่ไม่มี GPU
        _yolo.to("cpu")

    if _fusion_model is None:
        if not CKPT_FUSION:
            raise FileNotFoundError("Fusion checkpoint path/URL is not set (CKPT_FUSION or MODEL_URL).")
        _ensure_exists(CKPT_FUSION, "Fusion checkpoint")
        _fusion_model, _feat_names, _img_size_fusion = load_checkpoint(CKPT_FUSION, default_img_size=384)
        _fusion_model.to(torch.device("cpu")).eval()

    if _scaler is None and Path(SCALER_JSON).exists():
        try:
            with open(SCALER_JSON, "r", encoding="utf-8") as f:
                _scaler = json.load(f)
        except Exception:
            _scaler = None  # ใช้งานต่อได้แม้ไม่มี scaler


def predict_height_m(image_path: str) -> float:
    """
    รับ path รูป -> คืนความสูงรอยน้ำหน่วย 'เมตร' (ปัด 2 ตำแหน่ง)
    - ถ้าตรวจไม่เจอ waterline จะคืน NaN
    """
    try:
        _lazy_init()

        # ----- YOLO -----
        r = _yolo.predict(
            source=image_path,
            imgsz=IMG_SIZE_YOLO,
            conf=CONF_YOLO,
            iou=IOU_YOLO,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]
        H, W = r.orig_img.shape[:2]

        # สร้าง/ทำความสะอาด mask น้ำ
        wm = combine_water_masks(r, WATER_CLASS)
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)

        # หาเส้นน้ำ
        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        if y_curve is None:
            return float("nan")

        # แปลงเป็นพิกเซลความสูงจากล่างขึ้นบน
        pixel_height = (H - 1) - int(y_med)

        # อัตราส่วนอ้างอิงจากวัตถุอ้างอิง
        ref_dict = ref_sizes_from_result(r)

        # ฟีเจอร์
        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)

        im = load_image_tensor(image_path, _img_size_fusion).to(torch.device("cpu"))

        # ----- Fusion head -----
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=torch.device("cpu"))
            y = _fusion_model(im, X_t).detach().cpu().item()

        return float(r2([y], 2)[0])
    except Exception:
        # อย่าทำให้ทั้ง API ล่ม—คืน NaN แล้วไปแคลิเบรตต่อฝั่ง app
        return float("nan")


def predict_height_debug(image_path: str) -> dict:
    """
    รันเหมือน predict_height_m แต่คืน debug dict ละเอียดยิบ
    """
    dbg = {"ok": False, "reason": "", "y_raw": None}
    try:
        _lazy_init()
        dbg["device"] = "cpu" if not torch.cuda.is_available() else "cuda"
        dbg["weights_yolo"] = str(WEIGHTS_YOLO)
        dbg["ckpt_fusion"] = str(CKPT_FUSION)
        dbg["scaler_json"] = str(SCALER_JSON)
        dbg["scaler_loaded"] = bool(_scaler is not None)

        # ----- YOLO -----
        r = _yolo.predict(
            source=image_path,
            imgsz=IMG_SIZE_YOLO,
            conf=CONF_YOLO,
            iou=IOU_YOLO,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]
        H, W = r.orig_img.shape[:2]
        dbg["H"], dbg["W"] = int(H), int(W)

        # รายการกล่องที่เจอ
        boxes = []
        names = r.names if hasattr(r, "names") else {}
        if getattr(r, "boxes", None) is not None:
            for b in r.boxes:
                cls = int(b.cls.item())
                conf = float(b.conf.item()) if getattr(b, "conf", None) is not None else None
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append({"cls": cls, "name": names.get(cls, str(cls)),
                              "conf": round((conf or 0), 3), "xyxy": [x1, y1, x2, y2]})
        dbg["detections"] = boxes

        # ----- water mask / waterline -----
        wm = combine_water_masks(r, WATER_CLASS)
        dbg["water_pixels"] = int(wm.astype("uint8").sum())
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)
        dbg["water_pixels_clean"] = int(wm_clean.astype("uint8").sum())

        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        if y_curve is None:
            dbg["reason"] = "no_waterline_detected"
            return dbg

        pixel_height = (H - 1) - int(y_med)
        dbg["y_median"] = int(y_med)
        dbg["pixel_height"] = int(pixel_height)

        # ----- ref objects -----
        ref_dict = ref_sizes_from_result(r)
        dbg["ref_features"] = ref_dict

        # ----- features / scaler -----
        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)
        dbg["used_cols"] = list(used_cols)

        im = load_image_tensor(image_path, _img_size_fusion).to(torch.device("cpu"))
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=torch.device("cpu"))
            y = _fusion_model(im, X_t).detach().cpu().item()
        dbg["y_raw"] = float(y)
        dbg["y_m_rounded"] = float(r2([y], 2)[0])
        dbg["ok"] = True
        return dbg

    except Exception as e:
        import traceback
        dbg["reason"] = f"exception: {e}"
        dbg["trace"] = traceback.format_exc()
        return dbg


def infer_status() -> dict:
    """สถานะไฟล์/โมเดล สำหรับ health check ภายนอก"""
    try:
        info = {
            "weights_yolo": str(WEIGHTS_YOLO),
            "ckpt_fusion": str(CKPT_FUSION),
            "scaler_json": str(SCALER_JSON),
            "exists": {
                "weights_yolo": Path(WEIGHTS_YOLO).exists(),
                "ckpt_fusion": bool(CKPT_FUSION) and Path(CKPT_FUSION).exists(),
                "scaler_json": Path(SCALER_JSON).exists(),
            }
        }
        _lazy_init()
        info["model_ready"] = True
        return info
    except Exception as e:
        return {
            "model_ready": False,
            "error": str(e),
            "weights_yolo": str(WEIGHTS_YOLO),
            "ckpt_fusion": str(CKPT_FUSION),
            "scaler_json": str(SCALER_JSON),
        }
