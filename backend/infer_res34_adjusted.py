#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_oneclick_piecewise_auto.py  (all-in-one, 2-decimal, auto piecewise)

- YOLOv8-seg → สกัด waterline + วัตถุอ้างอิง (px)
- Fusion model (+scaler.json ถ้ามี) → ทำนายความสูงเป็นเมตร
- ปัดผล 2 ตำแหน่งเสมอ
- Auto piecewise calibration:
    * ถ้ามี GT: แบ่งช่วงด้วย error=(y_pred - y_true) ที่ p33/p66 แล้ว fit y_true=a*y_pred+b แยกช่วง
    * ถ้าไม่มี GT: ข้ามการปรับ (หรือเลือก --auto_piecewise by_pred เพื่อแบ่งช่วงตาม y_pred แต่ a=1,b=0)
- ประเมินกับ GT (ถ้ามี): MAE/RMSE/R² (RAW vs ADJ)
- เขียน CSV เดียว พร้อมคอลัมน์ adjust_bin, a_used, b_used

วิธีใช้ (ค่า default กำหนดให้แล้ว):
python infer_oneclick_piecewise_auto.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # workaround OpenMP duplicate (Windows)

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO

# ===== DEFAULT PATHS (แก้ให้ตรงเครื่องคุณ) =====
DEFAULT_WEIGHTS   = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\runs\segment\train_PinFlood_segment\weights\best.pt"
DEFAULT_IMAGE_DIR = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\Pin_Flood\remove_IQR_outlier_Pinflood_dataset\images\val"
DEFAULT_CKPT      = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\Pin_Flood\modelres34_out\fusion_resnet34_best.pt"  # ckpt ที่เทรนด้วย res34 (มี meta backbone ใน ckpt)
DEFAULT_SCALER    = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\Pin_Flood\modelres34_out\scaler.json"               # ไม่มีก็รันได้ (ข้าม scaling)
DEFAULT_OUT       = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\Pin_Flood\Prediction\result_modelmk1_res34\val2.2_predictions_Calib_scaled.csv"
DEFAULT_GT        = r"E:\499_cnx_flood_detect\YOLO_Flood_seg\Pin_Flood\All_Pin_Flood.csv"

WATER_CLASS = "Water_mark"
REF_CLASSES = ["Electric_pole","Brick","Motorcycle","Sedan_car","Pickup_car","SUV_car","Van_car","Pylon"]

# ---------- rounding helper ----------
def r2(x, n=2):
    return np.round(np.array(x, dtype=float), n)

# ---------- YOLO utils ----------
def combine_water_masks(result, target_class_name=WATER_CLASS):
    H, W = result.orig_img.shape[:2]
    names = result.names
    wm = np.zeros((H, W), np.uint8)
    if result.masks is None:
        return wm
    for i, cid in enumerate(result.boxes.cls.tolist()):
        if names[int(cid)] != target_class_name:
            continue
        m = result.masks.data[i].cpu().numpy()
        m = (m * 255).astype(np.uint8)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        wm = cv2.bitwise_or(wm, m)
    return wm

def clean_mask(mask, min_area=200, k_close=5, k_open=3):
    if mask is None or mask.max() == 0:
        return mask
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open,  k_open))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  ko)
    num, labs, stats, _ = cv2.connectedComponentsWithStats((m>0).astype(np.uint8), 8)
    out = np.zeros_like(m)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labs==i] = 255
    return out

def waterline_y_curve(mask_binary, smooth_px=15, min_points_ratio=0.02):
    H, W = mask_binary.shape
    y_top = np.full(W, np.nan, dtype=np.float32)
    for x in range(W):
        ys = np.where(mask_binary[:, x] > 0)[0]
        if len(ys) > 0:
            y_top[x] = ys.min()
    good = ~np.isnan(y_top)
    if good.sum() < max(5, int(min_points_ratio * W)):
        return None, None
    xs = np.arange(W, dtype=np.float32)
    y_interp = np.interp(xs, xs[good], y_top[good]).astype(np.float32)
    k = smooth_px if smooth_px % 2 == 1 else smooth_px + 1
    y_smooth = cv2.GaussianBlur(y_interp.reshape(1, -1), (k, 1), 0).ravel()
    y_curve = np.clip(np.round(y_smooth).astype(int), 0, H-1)
    y_median = int(np.median(y_curve))
    return y_curve, y_median

def ref_sizes_from_result(result):
    names = result.names
    H, W = result.orig_img.shape[:2]
    values = {c: [] for c in REF_CLASSES}
    if result.boxes is None:
        return {f"{k}_px": np.nan for k in REF_CLASSES}
    use_mask_bbox = result.masks is not None
    for i, cid in enumerate(result.boxes.cls.tolist()):
        cls = names[int(cid)]
        if cls not in REF_CLASSES:
            continue
        if use_mask_bbox and i < len(result.masks.data):
            mask = (result.masks.data[i].cpu().numpy()*255).astype(np.uint8)
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(mask>0)
            if len(xs)==0: continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        else:
            x0,y0,x1,y1 = result.boxes.xyxy[i].cpu().numpy().astype(int)
        width  = max(1, int(x1 - x0))
        height = max(1, int(y1 - y0))
        values[cls].append(width if cls=="Electric_pole" else height)
    out = {f"{k}_px": (float(np.mean(v)) if v else np.nan) for k, v in values.items()}
    return out

# ---------- Fusion model ----------
class FusionRegressor(nn.Module):
    def __init__(self, n_tab_features, backbone="res34"):
        super().__init__()
        if backbone == "res50":
            try:   self.cnn = models.resnet50(weights=None)
            except: self.cnn = models.resnet50(pretrained=False)
        elif backbone == "res18":
            try:   self.cnn = models.resnet18(weights=None)
            except: self.cnn = models.resnet18(pretrained=False)
        else:
            try:   self.cnn = models.resnet34(weights=None)
            except: self.cnn = models.resnet34(pretrained=False)
        in_feat = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        self.tab = nn.Sequential(
            nn.Linear(n_tab_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(in_feat + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, img, x_tab):
        f_img = self.cnn(img)
        f_tab = self.tab(x_tab)
        return self.head(torch.cat([f_img, f_tab], dim=1)).squeeze(1)

def load_checkpoint(ckpt_path, default_img_size=384):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    feat_names = ckpt.get("feature_names", None)
    img_size   = ckpt.get("img_size", default_img_size)
    backbone   = ckpt.get("backbone", "res34")
    model = FusionRegressor(n_tab_features=len(feat_names) if feat_names else 0, backbone=backbone)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, feat_names, img_size

# ---------- feature-engineering ----------
BASE_COLS = ["pixel_height","H","y_median","W"]
REF_COLS  = ["Electric_pole_px","Brick_px","Motorcycle_px","Sedan_car_px",
             "Pickup_car_px","SUV_car_px","Van_car_px","Pylon_px"]
PERSPECTIVE_COLS = ["water_area_norm","wm_conf_mean","pole_conf_mean"]

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    H  = out.get("H", pd.Series(np.nan, index=out.index)).astype(float)
    YM = out.get("y_median", pd.Series(np.nan, index=out.index)).astype(float)
    PX = out.get("pixel_height", pd.Series(np.nan, index=out.index)).astype(float)
    out["pixel_height_norm"] = PX / np.where(H>0, H, np.nan)
    out["y_median_norm"]     = YM / np.where(H>0, H, np.nan)
    pole = out.get("Electric_pole_px", pd.Series(np.nan, index=out.index)).astype(float)
    out["pole_ratio"] = np.where((PX>0)&(~np.isnan(PX)), pole/np.where(PX>0, PX, np.nan), np.nan)
    exist_refs = [c for c in REF_COLS if c in out.columns]
    out["n_refs"] = out[exist_refs].notna().sum(axis=1).astype(float) if exist_refs else 0.0
    return out

def apply_scaler_df(feats: pd.DataFrame, scaler: dict) -> pd.DataFrame:
    cols = scaler["feature_names"]
    X = feats.reindex(columns=cols)
    mean = np.array(scaler["mean"], dtype=np.float64)
    std  = np.array(scaler["std"],  dtype=np.float64)
    X_scaled = (X - mean) / np.where(std>1e-12, std, 1.0)
    return X_scaled

def make_feature_matrix(df, feature_names, scaler=None):
    df2 = add_engineered_features(df)
    # ถ้ามี scaler → ใช้มัน, ไม่มีก็ใช้ค่าดิบ (สมมติว่าโมเดลเทรนโดยไม่ scale)
    if scaler is not None:
        cols = scaler["feature_names"]
        feats = df2.reindex(columns=cols).copy()
        miss  = feats.isna().astype(float).add_prefix("isna_")
        feats = apply_scaler_df(feats, scaler).fillna(0.0)
        Xdf = pd.concat([feats, miss], axis=1)
    else:
        base = BASE_COLS + REF_COLS + ["pixel_height_norm","y_median_norm","pole_ratio","n_refs"]
        cols = [c for c in base if c in df2.columns]
        feats = df2[cols].copy()
        miss  = feats.isna().astype(float).add_prefix("isna_")
        Xdf = pd.concat([feats.fillna(0.0), miss], axis=1)

    for c in feature_names:
        if c not in Xdf.columns:
            Xdf[c] = 0.0
    Xdf = Xdf[feature_names]
    return Xdf.values.astype(np.float32), list(Xdf.columns)

def load_image_tensor(path, img_size):
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    im = Image.open(path).convert("RGB")
    return tf(im).unsqueeze(0)

def norm_key(s: str):
    stem = Path(str(s)).stem
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem

# ---------- YOLO feature builder ----------
def build_features_with_yolo(weights, image_dir, imgsz=960, conf=0.35, iou=0.5):
    model = YOLO(weights)
    images = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.tif","*.tiff"):
        images += list(Path(image_dir).glob(ext))
    images = sorted(images)

    rows = []
    for p in images:
        r = model.predict(source=str(p), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        H, W = r.orig_img.shape[:2]

        wm = combine_water_masks(r, WATER_CLASS)
        wm_clean = clean_mask(wm, min_area=200, k_close=5, k_open=3)
        y_curve, y_med = waterline_y_curve(wm_clean, smooth_px=15)
        found = y_curve is not None
        pixel_height = (H-1) - y_med if found else np.nan

        ref_dict = ref_sizes_from_result(r)

        row = {"image": p.name, "H": H, "W": W,
               "y_median": (int(y_med) if found else np.nan),
               "pixel_height": (int(pixel_height) if found else np.nan),
               "found": bool(found)}
        row.update(ref_dict)
        rows.append(row)
    return pd.DataFrame(rows)

# ---------- Piecewise helpers ----------
def fit_line(y_pred, y_true):
    """fit y_true ~ a*y_pred + b (least squares)"""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if len(y_pred) < 2:
        return 1.0, 0.0
    A = np.vstack([y_pred, np.ones_like(y_pred)]).T
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    return float(a), float(b)

def split_bins(values, mode="percentile", p_lo=33, p_hi=66):
    """คืนขอบช่วง (lo,hi) 3 ช่วงตาม percentiles"""
    v = np.asarray(values, dtype=float)
    lo = np.nanpercentile(v, p_lo)
    hi = np.nanpercentile(v, p_hi)
    return [(-1e18, lo), (lo, hi), (hi, 1e18)]

def assign_bin(val, bins):
    for i, (lo, hi) in enumerate(bins):
        if lo <= val < hi:
            return i
    return len(bins) - 1

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--images",  default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--ckpt",    default=DEFAULT_CKPT)
    ap.add_argument("--scaler",  default=DEFAULT_SCALER,
                    help="ถ้ามีจะใช้ normalize ฟีเจอร์ให้เหมือนตอนเทรน; ถ้าไม่มีจะข้าม scaling")
    ap.add_argument("--out",     default=DEFAULT_OUT)
    ap.add_argument("--imgsz",   type=int, default=960)
    ap.add_argument("--conf",    type=float, default=0.35)
    ap.add_argument("--iou",     type=float, default=0.5)
    ap.add_argument("--gt",      default=DEFAULT_GT)
    ap.add_argument("--auto_piecewise", default="auto",
                    choices=["auto","off","by_pred","by_error"],
                    help="auto: ใช้ by_error ถ้ามี GT, ไม่งั้น off")
    args = ap.parse_args()

    image_dir = Path(args.images)
    out_path  = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) YOLO → features
    print("🔍 Running YOLO & building features ...")
    df = build_features_with_yolo(args.weights, image_dir, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    mask_found = df["found"].astype(str).str.lower().isin(["1","true","yes"]) if "found" in df.columns else np.ones(len(df), bool)
    df_use = df[mask_found].copy()
    if df_use.empty:
        df.to_csv(out_path, index=False, encoding="utf-8-sig", na_rep="NaN")
        print("⚠️ ไม่พบรูปที่มี waterline → เซฟฟีเจอร์ดิบแทน"); print(f"✅ saved: {out_path}")
        return

    # 2) โหลด Fusion model + (optional) scaler
    print("🧠 Loading fusion model ...")
    model, feat_names, img_size = load_checkpoint(args.ckpt, default_img_size=384)
    scaler = None
    if args.scaler and Path(args.scaler).exists():
        try:
            scaler = json.load(open(args.scaler, "r", encoding="utf-8"))
            print("🧪 Loaded scaler.json")
        except Exception as e:
            print(f"⚠️ โหลด scaler.json ไม่สำเร็จ → ข้าม scaling ({e})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X, used_feature_names = make_feature_matrix(df_use, feature_names=feat_names, scaler=scaler)

    # 3) Predict (2-decimal)
    print("🚀 Predicting ...")
    preds = []
    for i, row in df_use.iterrows():
        img_name = row["image"]
        p = image_dir / img_name
        if not p.exists():
            stem = Path(img_name).stem
            for ext in [".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"]:
                cand = image_dir / f"{stem}{ext}"
                if cand.exists(): p = cand; break
        im = load_image_tensor(p, img_size).to(device)
        x = torch.tensor(X[df_use.index.get_loc(i)]).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(im, x).cpu().item()
        preds.append(y)
    preds = r2(preds, 2)

    # 4) DataFrame พื้นฐาน
    out_df = df_use[["image","H","W","y_median","pixel_height","found"] + [c for c in df_use.columns if c.endswith("_px")]].copy()
    out_df["pred_height_m_raw"] = preds.astype(float)

    # 5) เตรียม GT (ถ้ามี)
    gt_path = Path(args.gt) if args.gt else None
    merged = None
    if gt_path and gt_path.exists():
        try:
            df_gt = pd.read_csv(gt_path)
            key_col = next((c for c in ["image","record","filename","id","CM_code"] if c in df_gt.columns), None)
            if key_col is not None and "height_m_true" in df_gt.columns:
                tmp = out_df[["image","pred_height_m_raw"]].copy()
                tmp["key"] = tmp["image"].apply(norm_key)
                df_gt["key"] = df_gt[key_col].apply(norm_key)
                merged = tmp.merge(df_gt[["key","height_m_true"]], on="key", how="inner")
                if len(merged) == 0:
                    merged = None
        except Exception as e:
            print(f"⚠️ โหลด/เมิร์จ GT ไม่สำเร็จ: {e}")
            merged = None

    # 6) Auto piecewise
    mode = args.auto_piecewise
    if mode == "auto":
        mode = "by_error" if merged is not None else "off"

    did_adjust = False
    bin_names, a_used, b_used = None, None, None

    if mode == "by_error" and merged is not None:
        # ปัด GT 2-decimal เพื่อความสอดคล้อง
        merged["height_m_true"] = r2(merged["height_m_true"], 2)
        # คำนวณ error แล้วแบ่ง bins
        err = merged["pred_height_m_raw"].values - merged["height_m_true"].values
        bins = split_bins(err, p_lo=33, p_hi=66)  # 3 ช่วง
        # fit a,b ต่อช่วง
        a_list, b_list = [], []
        labels = []
        # map key -> (a,b) later
        merged["bin_id"] = [assign_bin(e, bins) for e in err]
        for k in range(3):
            mask = (merged["bin_id"].values == k)
            if mask.sum() >= 2:
                a,b = fit_line(merged.loc[mask,"pred_height_m_raw"], merged.loc[mask,"height_m_true"])
            else:
                a,b = 1.0, 0.0
            a_list.append(a); b_list.append(b)

        key2ab = {}
        for k in range(3):
            key2ab[k] = (a_list[k], b_list[k])

     
        key_to_bin = dict(zip(merged["key"].values, merged["bin_id"].values))
        out_df["key"] = out_df["image"].apply(norm_key)
        out_df["adjust_bin"] = out_df["key"].map(key_to_bin)
        a_used = out_df["adjust_bin"].map({i:a_list[i] for i in range(3)}).fillna(1.0).values
        b_used = out_df["adjust_bin"].map({i:b_list[i] for i in range(3)}).fillna(0.0).values

        out_df["pred_height_m"] = r2(a_used * out_df["pred_height_m_raw"].values + b_used, 2)
       
        out_df["adjust_bin"] = out_df["adjust_bin"].map({0:"Low",1:"Mid",2:"High"}).fillna("NA")
        bin_names = ["Low","Mid","High"]
        did_adjust = True
        print(f"🔧 Piecewise(by_error) applied; bins via error p33/p66. Params per bin:")
        for i,name in enumerate(bin_names):
            print(f"   - {name}: a={a_list[i]:.3f}, b={b_list[i]:.3f}")

    elif mode == "by_pred":
        # แบ่ง bins จาก y_pred เอง (ไม่มี GT ให้ fit → identity)
        y = out_df["pred_height_m_raw"].values
        bins = split_bins(y, p_lo=33, p_hi=66)
        out_df["adjust_bin"] = [assign_bin(v, bins) for v in y]
        out_df["pred_height_m"] = r2(out_df["pred_height_m_raw"].values, 2)  # a=1,b=0
        out_df["adjust_bin"] = out_df["adjust_bin"].map({0:"Low","1":"Mid","2":"High"}).fillna("NA")
        did_adjust = False  # no actual change
        print("ℹ️ by_pred mode (no GT): bins สร้างจาก y_pred แต่ไม่ปรับค่า (a=1,b=0)")

    else:
        out_df["pred_height_m"] = r2(out_df["pred_height_m_raw"], 2)
        print("ℹ️ piecewise: off")

    # 7) Save CSV (ปัดเลขสำคัญ)
    for c in ["pred_height_m_raw","pred_height_m"] + [c for c in out_df.columns if c.endswith("_px")]:
        if c in out_df.columns: out_df[c] = r2(out_df[c], 2)
    if a_used is not None:
        out_df["a_used"] = r2(a_used, 3)
        out_df["b_used"] = r2(b_used, 3)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig", na_rep="NaN")
    print(f"✅ saved: {out_path}")

    # 8) Metrics (ถ้ามี GT)
    if merged is not None:
        # รวม pred ที่ปรับแล้วกลับเข้า merged เพื่อคำนวณ metric
        # ทำ key ให้ unique ก่อน (เลือกใช้ค่าเฉลี่ยต่อ key)
        m_adj = (
            out_df[["key", "pred_height_m"]]
            .dropna()
            .groupby("key", as_index=True)["pred_height_m"]
            .mean()
        )
        m_raw = (
            out_df[["key", "pred_height_m_raw"]]
            .dropna()
            .groupby("key", as_index=True)["pred_height_m_raw"]
            .mean()
        )

        # ถ้าอยากใช้ค่าแรก/ล่าสุดแทนเฉลี่ย:
        # m_adj = (out_df.dropna(subset=["pred_height_m"])
        #                 .drop_duplicates("key", keep="last")
        #                 .set_index("key")["pred_height_m"])
        # m_raw = (out_df.dropna(subset=["pred_height_m_raw"])
        #                 .drop_duplicates("key", keep="last")
        #                 .set_index("key")["pred_height_m_raw"])

        merged["pred_height_m_adj"] = merged["key"].map(m_adj).astype(float)
        merged["pred_height_m_raw"] = merged["key"].map(m_raw).astype(float)

        def metric(y_true, y_pred):
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)
            mae  = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            ss_res = np.sum((y_true - y_pred)**2)
            ss_tot = np.sum((y_true - y_true.mean())**2)
            r2s = float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan
            return mae, rmse, r2s

        mae_r, rmse_r, r2_r = metric(merged["height_m_true"], merged["pred_height_m_raw"])
        mae_c, rmse_c, r2_c = metric(merged["height_m_true"], merged["pred_height_m_adj"])

        print(f"\n📏 Metrics on GT (matched {len(merged)} imgs) [2-decimal]:")
        print(f"  RAW : MAE={mae_r:.2f} m  RMSE={rmse_r:.2f} m  R^2={r2_r:.2f}")
        print(f"  ADJ : MAE={mae_c:.2f} m  RMSE={rmse_c:.2f} m  R^2={r2_c:.2f}")

        merged.to_csv(out_path.with_name(out_path.stem + "_with_gt.csv"),
                      index=False, encoding="utf-8-sig")
        print(f"📝 saved: {out_path.with_name(out_path.stem + '_with_gt.csv')}")
    else:
        print("ℹ️ ไม่มี/เมิร์จ GT ไม่ได้: แสดงได้เฉพาะผลทำนาย")


if __name__ == "__main__":
    main()
