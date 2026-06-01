"""Prediction utilities for the demo app."""

import re

import pandas as pd

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "account", "locked", "suspended", "click", "login",
    "password", "bank", "refund", "delivery", "package", "prize", "winner",
    "claim", "payment", "tax", "security", "limited time",
]

URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+?\d[\d\-\s()]{7,}\d)")


def detect_indicators(text: str) -> dict:
    """Detect URL, email, phone, and suspicious keywords in a message."""
    text_lower = text.lower()
    return {
        "has_url": bool(URL_PATTERN.search(text)),
        "has_email": bool(EMAIL_PATTERN.search(text)),
        "has_phone": bool(PHONE_PATTERN.search(text)),
        "suspicious_keywords": [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower],
    }


def prepare_input(text: str) -> pd.DataFrame:
    """Prepare a single SMS message as a DataFrame for model.predict()."""
    indicators = detect_indicators(text)
    text_clean = re.sub(r"\s+", " ", str(text).strip())
    return pd.DataFrame([{
        "text_clean": text_clean,
        "has_url": int(indicators["has_url"]),
        "has_email": int(indicators["has_email"]),
        "has_phone": int(indicators["has_phone"]),
    }])
