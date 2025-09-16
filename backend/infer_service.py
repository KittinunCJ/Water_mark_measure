# --- วางต่อท้ายไฟล์ infer_service.py ---

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
