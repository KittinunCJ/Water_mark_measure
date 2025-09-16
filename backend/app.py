# backend/app.py
# -*- coding: utf-8 -*-
import json, time, math
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# ====== Path setup ======
BASE_DIR = Path(__file__).resolve().parent            # .../backend
ROOT_DIR = BASE_DIR.parent                            # repo root
FRONTEND_DIR = ROOT_DIR / "frontend"                  # .../frontend
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "reports.jsonl"

# กันกรณี build/boot แรกที่ frontend ยังไม่มีไฟล์ -> อย่าทำให้แอปล่ม
if not (FRONTEND_DIR / "index.html").exists():
    FRONTEND_DIR = ROOT_DIR

# ====== Env (.env ได้ทั้งที่ root และ backend) ======
load_dotenv(ROOT_DIR / ".env")
load_dotenv(BASE_DIR / ".env")

# ====== JSON-safe helper ======
def _sanitize_for_json(obj):
    """แทน NaN/Inf ด้วย None เพื่อให้ json.dumps ทำงานได้"""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj

# ====== Calibration: y' = 0.9553*x + 0.0325 ======
def calibrate(level_m: float, ndigits: int = 2) -> float:
    if level_m is None or (isinstance(level_m, float) and math.isnan(level_m)):
        return level_m
    y = 0.9553 * float(level_m) + 0.0325
    return round(y, ndigits)

# ====== Inference loader ======
try:
    # uvicorn backend.app:app
    from .infer_service import predict_height_m
except Exception:
    try:
        # python backend/app.py
        from infer_service import predict_height_m
    except Exception:
        def predict_height_m(_):
            raise RuntimeError("ยังโหลดโมเดลไม่สำเร็จ (โปรดวางไฟล์ weights และปรับ ENV ให้ครบ)")

# ====== App ======
app = FastAPI(title="Flood Mark API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟไฟล์อัปโหลด (ไม่ชน /api/*)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ====== Helpers ======
def _save_record(rec: dict):
    rec = _sanitize_for_json(rec)
    with DB_FILE.open("a", encoding="utf-8") as f:
        # บังคับห้าม NaN/Inf ในไฟล์ถาวร
        f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n")

def _load_records(limit: int = 200) -> List[dict]:
    if not DB_FILE.exists():
        return []
    items = []
    with DB_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except:
                pass
    return items[-limit:]

# ====== API ======
@app.get("/api/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/api/reports")
def get_reports(limit: int = 200):
    items = _load_records(limit=limit)
    items = [_sanitize_for_json(x) for x in items]  # กัน record เก่า ๆ ที่มี NaN
    return {"ok": True, "items": items}

@app.post("/api/report")
async def create_report(
    image: UploadFile = File(...),
    lat: str = Form(...), lng: str = Form(...),
    date_iso: str = Form(...),
    object_type: str = Form(...),
    description: str = Form(""),
    address: str = Form(""),
    skip_infer: int = 0
):
    # 1) save file
    safe_name = f"{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    # 2) infer / mock
    if int(skip_infer or 0) == 1:
        water_level_m = 0.25
    else:
        try:
            water_level_m = predict_height_m(str(out_path))
        except Exception:
            water_level_m = float("nan")

    # 3) calibrate
    water_level_m = calibrate(water_level_m)

    rec = {
        "lat": float(lat), "lng": float(lng),
        "date_iso": date_iso,
        "object_type": object_type,
        "description": description,
        "address": address if address else f"{lat},{lng}",
        "photo_url": f"/uploads/{safe_name}",
        "water_level_m": water_level_m
    }
    rec = _sanitize_for_json(rec)  # << สำคัญ
    _save_record(rec)
    return JSONResponse({"ok": True, **rec})

# ====== Static frontend ======
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# For local run: python backend/app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=False)
