# app.py
# -*- coding: utf-8 -*-
import os, json, time, math
from pathlib import Path
from typing import List, Optional, Tuple, Any
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# ====== โหลดไฟล์ .env ======
from dotenv import load_dotenv

# ====== ตั้งค่าโฟลเดอร์ ======
BASE_DIR = Path(__file__).resolve().parent        # ที่เดียวกับ app.py (root ของ repo นี้)
FRONTEND_DIR = BASE_DIR.parent / "frontend"       # ถ้ามีโฟลเดอร์ frontend ใช้อันนี้ก่อน
if not (FRONTEND_DIR / "index.html").exists():    # ไม่มีก็ fallback มาเสิร์ฟจาก root (BASE_DIR)
    FRONTEND_DIR = BASE_DIR

UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR   = BASE_DIR / "data"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA_DIR / "reports.jsonl"

# โหลด environment จาก .env ที่อยู่ root/ข้าง ๆ app.py
load_dotenv(dotenv_path=BASE_DIR / ".env")

# ====== สมการปรับแก้ค่า (y' = 0.9553*x + 0.0325) ======
def calibrate(level_m: Optional[float], ndigits: int = 2) -> Optional[float]:
    """
    รับค่าเป็น 'เมตร' จากโมเดล แล้วปรับด้วยสมการ y' = 0.9553*x + 0.0325
    - ถ้าเป็น None/NaN จะคืน None (จะไม่บังคับเป็น 0 เพื่อเลี่ยงแสดงผล 0.00 แบบผิดความจริง)
    """
    if level_m is None:
        return None
    try:
        if isinstance(level_m, float) and math.isnan(level_m):
            return None
        y = 0.9553 * float(level_m) + 0.0325
        return round(y, ndigits)
    except Exception:
        return None

# ====== โหลดตัวทำนายจาก infer_service ======
# ถ้ายังไม่วางไฟล์ weights/ หรือ import ล้มเหลว จะตั้งฟังก์ชันจำลอง (mock) ให้คืน None
try:
    # ถ้ารันแบบแพ็กเกจ (เช่น uvicorn app:app)
    from infer_service import predict_height_m  # ไฟล์ infer_service.py อยู่ root เดียวกัน
except Exception as e:
    def predict_height_m(_image_path: str) -> Optional[float]:
        # mock – แจ้งว่าไม่ได้โหลดโมเดลจริง
        raise RuntimeError("ยังโหลดโมเดลไม่สำเร็จ (โปรดวางไฟล์ weights และปรับ ENV ให้ครบ)") from e

app = FastAPI(title="Flood Mark API (One-Platform)")

# CORS (จริง ๆ ถ้าเสิร์ฟโดเมนเดียวกัน/พาธสัมพัทธ์จะไม่ถูกใช้ แต่ใส่ไว้ไม่เสียหาย)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เสิร์ฟไฟล์อัปโหลด
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
# เสิร์ฟหน้าเว็บ / index.html และไฟล์สาธารณะ (รวมถึง cm_boundary.geojson) จาก FRONTEND_DIR
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ====== ฟังก์ชันช่วย ======
def _save_record(rec: dict) -> None:
    with DB_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _load_records(limit: int = 200) -> List[dict]:
    if not DB_FILE.exists():
        return []
    items: List[dict] = []
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
    # ส่งสถานะ + บอกว่าเสิร์ฟไฟล์เว็บจากที่ไหน (debug ช่วยเวลา deploy)
    return {
        "ok": True,
        "ts": int(time.time()),
        "frontend_dir": str(FRONTEND_DIR),
        "uploads_dir": str(UPLOAD_DIR),
        "data_dir": str(DATA_DIR),
    }

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
    skip_infer: int = 0  # 1 = ข้ามการ infer (โหมดทดสอบ)
):
    """
    อัปโหลดรูป -> (พยายาม) infer ระดับรอยน้ำ -> ปรับแก้ค่า -> บันทึกลงไฟล์ .jsonl
    ฝั่ง response จะส่งทั้งค่าก่อนปรับ (water_level_raw_m) และหลังปรับ (water_level_m)
    ถ้า infer ล้มเหลว จะคืน water_level_raw_m = None และแนบข้อความ infer_error
    """

    # 1) เซฟรูป
    safe_name = f"{int(time.time()*1000)}_{image.filename.replace(' ', '_')}"
    out_path = UPLOAD_DIR / safe_name
    out_path.write_bytes(await image.read())

    # 2) infer หรือ mock
    infer_error: Optional[str] = None
    water_level_raw_m: Optional[float] = None

    if int(skip_infer or 0) == 1:
        # โหมดทดสอบ: ใส่ค่าสมมติ (0.25 m) ให้เห็นสีบนแผนที่
        water_level_raw_m = 0.25
    else:
        try:
            val = predict_height_m(str(out_path))
            # รองรับกรณีบางโมเดลคืนเป็น NaN/None
            if val is None or (isinstance(val, float) and math.isnan(val)):
                water_level_raw_m = None
            else:
                water_level_raw_m = float(val)
        except Exception as e:
            # ถ้ารันโมเดลพลาด ให้บันทึกข้อความ error และไม่บังคับเป็นศูนย์
            infer_error = f"{type(e).__name__}: {e}"
            water_level_raw_m = None

    # 3) ใส่สมการปรับแก้ค่า (ถ้าไม่มีค่าดิบ จะได้ None)
    water_level_m = calibrate(water_level_raw_m)

    # 4) เตรียมเรคคอร์ดและบันทึก
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except Exception:
        return JSONResponse({"ok": False, "message": "lat/lng ไม่ถูกต้อง"}, status_code=400)

    rec = {
        "lat": lat_f,
        "lng": lng_f,
        "date_iso": date_iso,
        "object_type": object_type,
        "description": description,
        "address": address if address else f"{lat},{lng}",
        "photo_url": f"/uploads/{safe_name}",
        # ผล infer
        "water_level_raw_m": water_level_raw_m,  # ก่อนปรับสมการ (อาจเป็น None)
        "water_level_m": water_level_m,          # หลังปรับสมการ (อาจเป็น None)
    }
    if infer_error:
        rec["infer_error"] = infer_error

    _save_record(rec)

    # 5) ตอบกลับ
    return JSONResponse({"ok": True, **rec})


# ---- รันบนพอร์ต 7860 เวลาเรียกโดยตรง (เช่นบนแพลตฟอร์ม) ----
if __name__ == "__main__":
    import uvicorn
    # ถ้าคุณสั่งรันตรง ๆ: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
