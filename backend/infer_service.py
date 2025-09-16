# backend/infer_service.py
# -*- coding: utf-8 -*-
"""
ตัวห่อโมเดล:
- โหลด YOLO + Fusion model ครั้งเดียว แล้ว reuse
- รับ path รูป -> คืนค่าความสูงรอยน้ำ (เมตร, ปัด 2 ตำแหน่ง)
"""
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# ใช้ของเดิมจากไฟล์คุณ
from .infer_res34_adjusted import (
    WATER_CLASS, REF_CLASSES,
    combine_water_masks, clean_mask, waterline_y_curve, ref_sizes_from_result,
    FusionRegressor, load_checkpoint, add_engineered_features, apply_scaler_df,
    make_feature_matrix, load_image_tensor, r2
)

# ====== Path/Env ======
BASE_DIR       = Path(__file__).resolve().parent                  # .../backend
WEIGHTS_DIR    = BASE_DIR / "weights"

WEIGHTS_YOLO   = os.environ.get("WEIGHTS_YOLO", str(WEIGHTS_DIR / "best.pt"))
# ถ้า fusion checkpoint ของคุณชื่ออื่น ให้เปลี่ยนที่นี่
CKPT_FUSION    = os.environ.get("CKPT_FUSION",  str(WEIGHTS_DIR / "fusion_resnet34_best.pt"))
SCALER_JSON    = os.environ.get("SCALER_JSON",  str(WEIGHTS_DIR / "scaler.json"))

IMG_SIZE_YOLO  = int(os.environ.get("IMG_SIZE_YOLO", "960"))
CONF_YOLO      = float(os.environ.get("CONF_YOLO", "0.35"))
IOU_YOLO       = float(os.environ.get("IOU_YOLO",  "0.50"))

DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Lazy singletons ======
_yolo           = None
_fusion_model   = None
_feat_names     = None
_img_size_fusion = 384
_scaler         = None


def _ensure_exists(p: str | Path, label: str):
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return str(p)


def _lazy_init():
    """โหลดโมเดล/สเกลเลอร์ครั้งเดียว"""
    global _yolo, _fusion_model, _feat_names, _img_size_fusion, _scaler

    if _yolo is None:
        _ensure_exists(WEIGHTS_YOLO, "YOLO weights")
        # บอก device ชัด ๆ (ultralytics จะเลือกเองได้ แต่กำหนดไว้กันพลาด)
        _yolo = YOLO(WEIGHTS_YOLO)
        _yolo.to(str(DEVICE))

    if _fusion_model is None:
        # ถ้าไม่มี ckpt ให้พยายามโหลดจาก path env—ถ้าไม่มีก็ให้ error อ่านง่าย
        _ensure_exists(CKPT_FUSION, "Fusion checkpoint")
        _fusion_model, _feat_names, _img_size_fusion = load_checkpoint(
            CKPT_FUSION, default_img_size=384
        )
        _fusion_model.to(DEVICE).eval()

    if _scaler is None and Path(SCALER_JSON).exists():
        try:
            with open(SCALER_JSON, "r", encoding="utf-8") as f:
                _scaler = json.load(f)
        except Exception:
            _scaler = None  # ใช้แบบไม่มี scaler ได้


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

        # อัตราส่วนอ้างอิงจากวัตถุอ้างอิง (เสา/ป้าย ฯลฯ) ที่ YOLO เห็น
        ref_dict = ref_sizes_from_result(r)

        # สร้าง dataframe ฟีเจอร์
        df = pd.DataFrame([{
            "image": Path(image_path).name,
            "H": H, "W": W,
            "y_median": int(y_med),
            "pixel_height": int(pixel_height),
            **ref_dict
        }])

        # สร้างเมทริกซ์ฟีเจอร์ตามลิสต์ที่เช็คพอยต์ต้องการ + สเกลถ้ามี
        X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)

        # เตรียมรูปสำหรับสาขา image ของ fusion
        im = load_image_tensor(image_path, _img_size_fusion).to(DEVICE)

        # ----- Fusion head -----
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
            y = _fusion_model(im, X_t).detach().cpu().item()

        # ปัดสองตำแหน่ง (ในโค้ดคุณมี r2 ก็ใช้ต่อได้)
        return float(r2([y], 2)[0])
    except Exception:
        # อย่าทำให้ทั้ง API ล่ม—คืน NaN แล้วไปแคลิบเรตต่อฝั่ง app
        return float("nan")

def predict_height_debug(image_path: str) -> dict:
    """
    รันเหมือน predict_height_m แต่คืน debug dict ละเอียดยิบ
    """
    dbg = {"ok": False, "reason": "", "y_raw": None}
    try:
        _lazy_init()
        dbg["device"] = DEVICE.type
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
                boxes.append({"cls": cls, "name": names.get(cls, str(cls)), "conf": round(conf or 0, 3),
                              "xyxy": [x1, y1, x2, y2]})
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

        im = load_image_tensor(image_path, _img_size_fusion).to(DEVICE)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
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

