# ocr_server.py — OCR Adapter unificado (RapidOCR / PaddleOCR / Tesseract)
# ------------------------------------------------------------
# Motores soportados: rapidocr | paddle | tesseract (selección por .env)
# Endpoints:
#   GET  /health   -> estado de motores y entorno
#   POST /ocr      -> {"image_b64": "..."} -> {"text": "...", "confidence": 0..1|null}
# ------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Tuple, List
import base64, io, os, sys, platform, logging
from PIL import Image
import numpy as np
# --- cargar .env muy temprano ---
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


# ---------------------------
# Configuración / Logs
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ocr-adapter")

ENGINE = os.getenv("OCR_SELECTED", os.getenv("OCR_ENGINE_NAME", "rapidocr")).strip().lower()
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()  # opcional (ruta en Windows)
MAX_WH = int(os.getenv("OCR_MAX_WH", "3000"))  # límite de tamaño en px

# ---------------------------
# Cargas perezosas por motor
# ---------------------------
_rapid = None; _rapid_err = None
_paddle = None; _paddle_err = None
_has_tess = False; _tess_err = None

def _load_rapid():
    global _rapid, _rapid_err
    if _rapid is not None or _rapid_err is not None:
        return
    try:
        from rapidocr_onnxruntime import RapidOCR
        _rapid = RapidOCR()
        log.info("RapidOCR cargado correctamente.")
    except Exception as e:
        _rapid_err = repr(e)
        log.warning(f"No se pudo cargar RapidOCR: {_rapid_err}")

def _load_paddle():
    global _paddle, _paddle_err
    if _paddle is not None or _paddle_err is not None:
        return
    try:
        from paddleocr import PaddleOCR
        _paddle = PaddleOCR(use_angle_cls=True, lang='es')  # soporte español
        log.info("PaddleOCR cargado correctamente (lang=es).")
    except Exception as e:
        _paddle_err = repr(e)
        log.warning(f"No se pudo cargar PaddleOCR: {_paddle_err}")

def _load_tesseract():
    global _has_tess, _tess_err
    if _has_tess or _tess_err is not None:
        return
    try:
        import pytesseract
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        _has_tess = True
        log.info("Tesseract disponible.")
    except Exception as e:
        _tess_err = repr(e)
        log.warning(f"No se pudo inicializar Tesseract: {_tess_err}")

# ---------------------------
# FastAPI
# ---------------------------
app = FastAPI(title="OCR Adapter — RapidOCR / PaddleOCR / Tesseract")

class OCRIn(BaseModel):
    image_b64: str

class OCROut(BaseModel):
    text: str
    confidence: Optional[float] = None

# ---------------------------
# Utilidades
# ---------------------------
def _img_from_b64(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def _maybe_resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= MAX_WH:
        return img
    scale = MAX_WH / float(m)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size)

# ---------------------------
# Wrappers de cada OCR
# ---------------------------
def _ocr_rapid(img: Image.Image) -> Tuple[str, Optional[float]]:
    _load_rapid()
    if _rapid is None:
        raise RuntimeError(f"RapidOCR no disponible: {_rapid_err}")
    arr = np.array(img)
    result, _ = _rapid(arr)
    if not result:
        return "", None
    lines, scores = [], []
    for item in result:
        if len(item) >= 3:
            _, txt, sc = item
        else:
            txt = item[1] if len(item) > 1 else ""
            sc = None
        if txt:
            lines.append(txt)
        if isinstance(sc, (int, float)):
            scores.append(float(sc))
    text = "\n".join(lines).strip()
    conf = round(sum(scores)/len(scores), 4) if scores else None
    return text, conf

def _ocr_paddle(img: Image.Image) -> Tuple[str, Optional[float]]:
    _load_paddle()
    if _paddle is None:
        raise RuntimeError(f"PaddleOCR no disponible: {_paddle_err}")
    arr = np.array(img)
    result = _paddle.ocr(arr, cls=True)
    lines, scores = [], []
    for page in result:
        for line in page:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                txt, sc = line[1]
                lines.append(txt)
                if isinstance(sc, (int, float)):
                    scores.append(float(sc))
    text = "\n".join(lines).strip()
    conf = round(sum(scores)/len(scores), 4) if scores else None
    return text, conf

def _ocr_tesseract(img: Image.Image) -> Tuple[str, Optional[float]]:
    _load_tesseract()
    if not _has_tess:
        raise RuntimeError(f"Tesseract no disponible: {_tess_err}")
    import pytesseract
    text = pytesseract.image_to_string(img)
    return (text or "").strip(), None

def _run_with_fallback(img: Image.Image, primary: str) -> Tuple[str, Optional[float], str]:
    order: List[str] = []
    if primary in ("rapidocr", "paddle", "tesseract"):
        order.append(primary)
    for alt in ("rapidocr", "paddle", "tesseract"):
        if alt not in order:
            order.append(alt)

    last_err = None
    for engine in order:
        try:
            if engine == "rapidocr":
                t, c = _ocr_rapid(img)
            elif engine == "paddle":
                t, c = _ocr_paddle(img)
            else:
                t, c = _ocr_tesseract(img)
            return t, c, engine
        except Exception as e:
            last_err = f"{engine}: {e!r}"
            log.warning(f"OCR con {engine} falló: {last_err}")
            continue
    raise RuntimeError(f"Todos los OCR fallaron. Último error: {last_err}")

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    try: _load_rapid()
    except: pass
    try: _load_paddle()
    except: pass
    try: _load_tesseract()
    except: pass
    return {
        "engine_requested": ENGINE,
        "rapidocr_loaded": _rapid is not None,
        "rapidocr_err": _rapid_err,
        "paddle_loaded": _paddle is not None,
        "paddle_err": _paddle_err,
        "tesseract_available": _has_tess,
        "tesseract_err": _tess_err,
        "py": sys.version,
        "os": platform.platform(),
        "max_wh_px": MAX_WH
    }

@app.post("/ocr", response_model=OCROut)
def ocr_endpoint(body: OCRIn):
    try:
        img = _img_from_b64(body.image_b64)
        img = _maybe_resize(img)
        text, conf, used = _run_with_fallback(img, ENGINE)
        log.info(f"OCR OK engine={used} len={len(text)} conf={conf}")
        return OCROut(text=text, confidence=conf)
    except Exception as e:
        log.exception("OCR error")
        raise HTTPException(status_code=500, detail=str(e))
