"""FastAPI demo app for SMS smishing classification."""

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.models.predict import detect_indicators, prepare_input

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="SMS Smishing Detector")

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MODEL_PATH = Path(__file__).resolve().parents[2] / "outputs" / "models" / "best_model.joblib"
META_PATH = MODEL_PATH.parent / "best_model_metadata.json"

model = None
metadata = None
LABELS = ["ham", "spam", "smishing"]

SAMPLE_MESSAGES = [
    {"label": "Ham (casual)", "text": "Hey! Are we still on for lunch tomorrow? Let me know."},
    {"label": "Ham (reminder)", "text": "Reminder: Your dentist appointment is scheduled for Monday at 2:30 PM."},
    {"label": "Spam", "text": "CONGRATULATIONS! You've been selected for a FREE cruise vacation! Call now to claim your spot. Limited availability!"},
    {"label": "Smishing (bank)", "text": "ALERT: Your bank account has been locked due to suspicious activity. Verify your identity now at http://secure-bank-login.com/verify"},
    {"label": "Smishing (package)", "text": "Your package delivery failed. Update your address and pay the $1.99 redelivery fee here: http://track-shipment.info/pay"},
    {"label": "Smishing (prize)", "text": "You've won a $500 gift card! Claim your prize by entering your details at http://rewards-center.net/claim before it expires."},
]


@app.on_event("startup")
def load_model():
    global model, metadata
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    if META_PATH.exists():
        with open(META_PATH) as f:
            metadata = json.load(f)


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------


def classify(text: str) -> dict:
    """Run prediction and return structured result."""
    indicators = detect_indicators(text)
    input_df = prepare_input(text)

    prediction = model.predict(input_df)[0]

    probabilities = None
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        class_labels = model.classes_
        probabilities = {label: round(float(p), 4) for label, p in zip(class_labels, proba)}
        confidence = round(float(max(proba)), 4)

    return {
        "message": text,
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "indicators": indicators,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "sample_messages": SAMPLE_MESSAGES,
        "result": None,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict_form(request: Request, message: str = Form(...)):
    result = classify(message)
    return templates.TemplateResponse(request, "index.html", {
        "sample_messages": SAMPLE_MESSAGES,
        "result": result,
    })


@app.post("/api/predict")
async def predict_api(request: Request):
    body = await request.json()
    text = body.get("message", "")
    if not text.strip():
        return JSONResponse({"error": "message is required"}, status_code=400)
    return JSONResponse(classify(text))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}
