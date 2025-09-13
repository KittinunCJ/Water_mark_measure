# backend/infer_service.py
# -*- coding: utf-8 -*-
"""
ตัวห่อโมเดลของคุณ (ใช้ infer_res34_adjusted.py เดิม)
- โหลด YOLO + Fusion model ครั้งเดียว แล้ว reuse
- คืนค่าความสูงรอยน้ำ (เมตร, ปัด 2 ตำแหน่ง)
"""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# นำฟังก์ชัน/คลาสจากไฟล์ของคุณมาใช้โดยตรง
from .infer_res34_adjusted import (
    WATER_CLASS, REF_CLASSES,
    combine_water_masks, clean_mask, waterline_y_curve, ref_sizes_from_result,
    FusionRegressor, load_checkpoint, add_engineered_features, apply_scaler_df,
    make_feature_matrix, load_image_tensor, r2
)

# ====== กำหนดไฟล์โมเดลผ่าน ENV (ตั้งในแพลตฟอร์ม) ======
WEIGHTS_YOLO = os.environ.get("WEIGHTS_YOLO", "backend/weights/best.pt")
CKPT_FUSION  = os.environ.get("CKPT_FUSION",  "backend/weights/fusion_resnet34_best.pt")
SCALER_JSON  = os.environ.get("SCALER_JSON",  "backend/weights/scaler.json")  # (ถ้ามี)

IMG_SIZE_YOLO = int(os.environ.get("IMG_SIZE_YOLO", "960"))
CONF_YOLO     = float(os.environ.get("CONF_YOLO", "0.35"))
IOU_YOLO      = float(os.environ.get("IOU_YOLO",  "0.5"))

_device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_yolo         = None
_model        = None
_feat_names   = None
_img_size_fusion = 384
_scaler       = None

def _lazy_init():
    global _yolo, _model, _feat_names, _img_size_fusion, _scaler
    if _yolo is None:
        _yolo = YOLO(WEIGHTS_YOLO)
    if _model is None:
        _model, _feat_names, _img_size_fusion = load_checkpoint(CKPT_FUSION, default_img_size=384)
        _model.to(_device).eval()
    if _scaler is None and Path(SCALER_JSON).exists():
        try:
            _scaler = json.load(open(SCALER_JSON, "r", encoding="utf-8"))
        except Exception:
            _scaler = None

def predict_height_m(image_path: str) -> float:
    """รับ path รูป, คืนความสูงรอยน้ำหน่วย 'เมตร' (ปัด 2 ตำแหน่ง)"""
    _lazy_init()
    r = _yolo.predict(source=image_path, imgsz=IMG_SIZE_YOLO, conf=CONF_YOLO, iou=IOU_YOLO, verbose=False)[0]
    H, W = r.orig_img.shape[:2]

    wm = combine_water_masks(r, WATER_CLASS)
    wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)

    y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
    if y_curve is None:
        return float("nan")

    pixel_height = (H - 1) - int(y_med)
    ref_dict = ref_sizes_from_result(r)

    df = pd.DataFrame([{
        "image": Path(image_path).name, "H": H, "W": W,
        "y_median": int(y_med), "pixel_height": int(pixel_height), **ref_dict
    }])

    # features → model
    X, used_cols = make_feature_matrix(df, feature_names=_feat_names, scaler=_scaler)
    im = load_image_tensor(image_path, _img_size_fusion).to(_device)
    with torch.no_grad():
        y = _model(im, torch.tensor(X).to(_device)).cpu().item()
    return float(r2([y], 2)[0])  # ปัด 2 ตำแหน่ง
