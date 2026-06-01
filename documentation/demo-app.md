# Demo Application

## Source: `src/app/main.py`, `src/app/templates/index.html`, `src/models/predict.py`

## Purpose

A locally-deployed web application that lets users paste SMS messages and receive classification predictions (ham/spam/smishing) using the best trained model.

## Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Templates | Jinja2 |
| Server | Uvicorn |
| Container | Docker (python:3.11-slim) |
| Model | joblib-serialized scikit-learn Pipeline |

## Endpoints

### `GET /`

Renders the HTML form with:
- Textarea for pasting SMS text
- Submit button
- Expandable section with 6 sample messages (clickable, auto-submit)

### `POST /predict`

Form submission endpoint. Accepts `message` form field. Returns the same HTML page with classification results displayed.

### `POST /api/predict`

JSON API endpoint.

**Request:**
```json
{"message": "Your account has been locked. Verify at http://example.com"}
```

**Response:**
```json
{
  "message": "Your account has been locked. Verify at http://example.com",
  "prediction": "smishing",
  "confidence": 0.9679,
  "probabilities": {"ham": 0.0007, "smishing": 0.9679, "spam": 0.0314},
  "indicators": {
    "has_url": true,
    "has_email": false,
    "has_phone": false,
    "suspicious_keywords": ["verify", "account", "locked"]
  }
}
```

### `GET /health`

Returns `{"status": "ok", "model_loaded": true}`.

## Prediction Flow

1. **Input preparation** (`predict.py:prepare_input`): Create a single-row DataFrame with `text_clean`, `has_url`, `has_email`, `has_phone`
2. **Indicator detection** (`predict.py:detect_indicators`): Regex-based URL/email/phone detection + keyword matching
3. **Model prediction**: `model.predict(df)` returns class label
4. **Probability extraction**: `model.predict_proba(df)` returns calibrated probabilities, aligned using `model.classes_`
5. **Response assembly**: Combine prediction, confidence, probabilities, and indicators

## Indicator Detection

### Pattern Matching

| Indicator | Pattern |
|-----------|---------|
| URL | `https?://` or `www.` |
| Email | Standard email regex |
| Phone | `+?\d[\d\-\s()]{7,}\d` |

### Suspicious Keywords

```
urgent, verify, account, locked, suspended, click, login, password,
bank, refund, delivery, package, prize, winner, claim, payment, tax,
security, limited time
```

These are **heuristic signals** displayed alongside the ML prediction. They are not used by the model itself — they help users understand why a message might be suspicious.

## Sample Messages

Six pre-loaded examples in the UI:

1. Ham (casual): "Hey! Are we still on for lunch tomorrow?"
2. Ham (reminder): "Reminder: Your dentist appointment is scheduled for Monday at 2:30 PM."
3. Spam: "CONGRATULATIONS! You've been selected for a FREE cruise vacation!"
4. Smishing (bank): "ALERT: Your bank account has been locked..."
5. Smishing (package): "Your package delivery failed. Update your address..."
6. Smishing (prize): "You've won a $500 gift card! Claim your prize..."

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY outputs/models/ outputs/models/
EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
services:
  smishing-demo:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs/models:/app/outputs/models
```

The volume mount allows updating the model without rebuilding the image.

### Commands

```bash
docker compose up --build      # Build and start
docker compose up -d           # Start detached
docker compose down            # Stop and remove
```

App available at: `http://localhost:8000`

## Known Quirks

- **Starlette TemplateResponse signature**: Newer versions of Starlette/FastAPI require `TemplateResponse(request, name, context)` instead of the older `TemplateResponse(name, {"request": request, ...})` pattern.
- **Model class order**: The fitted model's `.classes_` is `['ham', 'smishing', 'spam']` (alphabetical), not the display order. The app uses `model.classes_` to correctly align probability values with labels.
- **No LLM explanations**: The app uses only deterministic indicators. This is by design per the requirements.
