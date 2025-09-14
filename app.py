# backend/app.py
# -*- coding: utf-8 -*-
import os, json, time, math
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

from dotenv import load_dotenv

# ====== ตั้งค่าโฟลเดอร์ ======
BASE_DIR = Path(__file__).resolve().parent      # .../backend
FRONTEND_DIR = BASE_DIR.parent                  # root ของ repo (มี index.html)
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "reports.jsonl"

# fallback เผื่อย้ายไฟล์ในอนาคต
if not (FRONTEND_DIR / "index.html").exists():
    FRONTEND_DIR = BASE_DIR

# โหลด environment จาก backend/.env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ====== Utilities ======
def _is_bad_number(x: float) -> bool:
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))

# สมการปรับแก้ค่า (y' = 0.9553*x + 0.0325) + กัน NaN/Inf
def calibrate(level_m: float, ndigits: int = 2):
    if level_m is None or _is_bad_number(level_m):
        return None
    y = 0.9553 * float(level_m) + 0.0325
    if _is_bad_number(y):
        return None
    return round(y, ndigits)

# ====== โหลดตัวทำนายจาก infer_service ======
# (ตรง ๆ ตามของเดิม; ถ้าลงโมเดลไม่ครบจะ raise ในตอนเรียกใช้งาน)
try:
    from .infer_service import predict_height_m
except Exception:
    try:
        from infer_service import predict_height_m
    except Exception:
        def predict_height_m(_):
            raise RuntimeError("ยังโหลดโมเดลไม่สำเร็จ (โปรดวางไฟล์ weights และปรับ ENV ให้ครบ)")

app = FastAPI(title="Flood Mark API (One-Platform)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟไฟล์อัปโหลด (ไม่ชน /api/*)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# ====== ฟังก์ชันช่วย ======
def _save_record(rec: dict):
    with DB_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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

# ====== API routes ======
@app.get("/api/health")
def health():
    return {"ok": True, "ts": int(time.time())}

@app.get("/api/reports")
def get_reports(limit: int = 200):
    return {"ok": True, "items": _load_records(limit=limit)}

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
    # 1) เซฟรูป
    safe_name = f"{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    # 2) infer หรือ mock
    if int(skip_infer or 0) == 1:
        water_level_m = 0.25
    else:
        try:
            water_level_m = predict_height_m(str(out_path))
        except Exception:
            water_level_m = None  # ❗ อย่าใช้ float("nan")

    # 3) ปรับแก้ค่าและกัน NaN/Inf
    water_level_m = calibrate(water_level_m)
    if _is_bad_number(water_level_m):
        water_level_m = None

    rec = {
        "lat": float(lat), "lng": float(lng),
        "date_iso": date_iso,
        "object_type": object_type,
        "description": description,
        "address": address if address else f"{lat},{lng}",
        "photo_url": f"/uploads/{safe_name}",
        "water_level_m": water_level_m
    }
    _save_record(rec)
    return JSONResponse({"ok": True, **rec})

# ====== หน้าเว็บหลัก (เสิร์ฟตรง ๆ ให้เบา/เสถียร) ======
@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# ถ้ามีโฟลเดอร์ static จริงใน root ให้เสิร์ฟเพิ่มได้
STATIC_DIR = FRONTEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---- รันบนพอร์ต 7860 เวลาเรียกตรง ๆ ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=False)
