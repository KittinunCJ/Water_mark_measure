# backend/app.py
# -*- coding: utf-8 -*-
import os, json, time, math
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ====== โหลดไฟล์ .env ======
from dotenv import load_dotenv

# ====== ตั้งค่าโฟลเดอร์ ======
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "reports.jsonl"

# โหลด environment จาก backend/.env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ====== สมการปรับแก้ค่า (y' = 0.9553*x + 0.0325) ======
def calibrate(level_m: float, ndigits: int = 2) -> float:
    """
    รับค่าเป็น 'เมตร' จากโมเดล แล้วปรับด้วยสมการ y' = 0.9553*x + 0.0325
    - ถ้าเป็น NaN/None จะคืนค่าเดิม
    """
    if level_m is None or (isinstance(level_m, float) and math.isnan(level_m)):
        return level_m
    y = 0.9553 * float(level_m) + 0.0325
    return round(y, ndigits)

# ====== โหลดตัวทำนายจาก infer_service ======
# ถ้ายังไม่วางไฟล์ weights/ ให้ใช้โหมดทดสอบ (skip_infer=1)
try:
    # กรณีรันเป็นแพ็กเกจ (เช่น uvicorn backend.app:app)
    from .infer_service import predict_height_m
except Exception:
    try:
        # กรณีรันไฟล์ตรง ๆ (python app.py)
        from infer_service import predict_height_m
    except Exception:
        def predict_height_m(_):
            raise RuntimeError("ยังโหลดโมเดลไม่สำเร็จ (โปรดวางไฟล์ weights และปรับ ENV ให้ครบ)")

app = FastAPI(title="Flood Mark API (One-Platform)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # แม้จะ same-origin อยู่แล้ว แต่เปิดไว้ไม่เสียหาย
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟไฟล์อัปโหลด (ปล่อยไว้ต้นไฟล์ได้ เพราะไม่ชน /api/*)
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
    skip_infer: int = 0  # ⟵ รองรับโหมดทดสอบ (ไม่รันโมเดล)
):
    # 1) เซฟรูป
    safe_name = f"{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    # 2) infer หรือ mock
    if int(skip_infer or 0) == 1:
        # โหมดทดสอบ: ใส่ค่าสมมติ (0.25 m) ให้เห็นสีบนแผนที่
        water_level_m = 0.25
    else:
        try:
            water_level_m = predict_height_m(str(out_path))
        except Exception:
            # ถ้ารันโมเดลพลาด ให้เซฟ record ไว้อยู่ดี (water_level_m = NaN)
            water_level_m = float("nan")

    # 3) ✅ ใส่สมการปรับแก้ค่า y' = 0.9553*x + 0.0325
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
    _save_record(rec)
    return JSONResponse({"ok": True, **rec})

# ✅ ย้ายการเสิร์ฟหน้าเว็บมา “ท้ายไฟล์” เพื่อไม่ให้ดัก /api/*
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ---- รันบนพอร์ต 7860 เวลาเรียกโดยตรง (เช่นบนแพลตฟอร์ม) ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=False)
