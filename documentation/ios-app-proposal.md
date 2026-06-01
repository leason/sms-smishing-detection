# iOS SMS Filtering App — Proposal

## Overview

This document proposes an iOS application that uses the smishing detection model from this project to filter incoming SMS messages from unknown senders. The app would classify messages in real-time on-device and log results for user review.

This is a **stretch goal** — the core project is complete without it.

## Concept

```
┌──────────────────────┐      ┌──────────────────────┐
│  Message Filter       │      │  Companion App        │
│  Extension            │      │                       │
│                       │      │  ┌─────────────────┐  │
│  Receives unknown-    │      │  │ Message Log      │  │
│  sender SMS from iOS  │─────▶│  │                 │  │
│                       │      │  │ - Date/time     │  │
│  Classifies on-device │      │  │ - Prediction    │  │
│  as allow/junk/promo  │      │  │ - Confidence    │  │
│                       │      │  │ - Indicators    │  │
│  Writes to shared     │      │  └─────────────────┘  │
│  App Group storage    │      │                       │
└──────────────────────┘      │  Filter settings       │
                               │  Model info            │
                               └──────────────────────┘
```

## How iOS Message Filtering Works

Apple provides `ILMessageFilterExtension` (IdentityLookup framework) for SMS/MMS filtering:

1. When a message arrives from an **unknown sender** (not in Contacts), iOS passes it to the filter extension
2. The extension classifies it as `.allow`, `.junk`, or `.promotion` (iOS 16+ adds subcategories like `.transaction`, `.promotion`)
3. iOS moves junk messages to a separate "Junk" tab in Messages
4. The extension runs in a **sandboxed process** with limited resources

### Key Constraints

| Constraint | Detail |
|------------|--------|
| **Unknown senders only** | Messages from known contacts are never filtered |
| **Memory limit** | Extension gets ~6 MB of memory |
| **No network (offline mode)** | On-device classification has no network access |
| **Network mode available** | Can defer to a server, but adds latency and privacy concerns |
| **No UI in extension** | Extension is headless; companion app provides the UI |
| **Apple review** | Must not exfiltrate message content; must provide clear utility |

## Architecture

### Two Targets

1. **Message Filter Extension** — `ILMessageFilterExtension` subclass that classifies SMS
2. **Companion App** — SwiftUI app for browsing filtered messages and managing settings

### Shared Storage

Both targets join the same **App Group**, giving them a shared container directory. Filtered message metadata is stored in a shared SQLite database (via Core Data or GRDB).

### Data Model

```swift
struct FilteredMessage {
    let id: UUID
    let text: String
    let sender: String
    let receivedAt: Date
    let prediction: String        // "ham", "spam", "smishing"
    let confidence: Double?
    let filterAction: String      // "allow", "junk", "promotion"
    let hasURL: Bool
    let hasEmail: Bool
    let hasPhone: Bool
    let suspiciousKeywords: [String]
}
```

## Model Conversion — The Key Challenge

Our current model is a scikit-learn Pipeline serialized as `.joblib`. This cannot run on iOS. There are three viable paths:

### Option A: Core ML Conversion (preferred if feasible)

Use `coremltools` to convert the sklearn pipeline to a `.mlmodel` file.

**Pros:**
- Uses our exact trained model
- Core ML is optimized for on-device inference
- Integrates natively with Swift

**Challenges:**
- TF-IDF vectorizers with large vocabularies (50k+ features with bigrams) are tricky to convert
- The `ColumnTransformer` + `CalibratedClassifierCV` wrapper adds complexity
- Memory footprint of the vocabulary may exceed extension limits

**Feasibility test:**
```python
import coremltools as ct
model = joblib.load("outputs/models/best_model.joblib")
coreml_model = ct.converters.sklearn.convert(model, ...)
```

If the vocabulary is too large, we could retrain with reduced features (e.g., `max_features=5000`) specifically for the iOS model.

### Option B: Apple Create ML Text Classifier

Train a fresh model using Apple's Create ML framework directly on our cleaned dataset.

**Pros:**
- Designed for on-device text classification
- Handles tokenization internally
- Small model footprint
- Native Swift integration

**Cons:**
- Different model architecture than our research models
- May not match our sklearn F1 scores exactly
- Requires macOS + Xcode for training

**Process:**
1. Export `manual_clean.csv` (or combined) in Create ML format
2. Train `MLTextClassifier` in Xcode or via Swift script
3. Produces a `.mlmodel` file ready for embedding

### Option C: Lightweight Heuristic Classifier

Skip ML entirely; implement a rule-based classifier in Swift.

**Pros:**
- Tiny footprint, no model file
- Deterministic and explainable
- Easy to implement and review

**Cons:**
- Lower accuracy than ML approach
- Rigid — doesn't generalize beyond the rules

**Implementation:**
- Score messages based on: URL presence, suspicious keywords, phone number patterns, urgency indicators
- Threshold-based classification

### Recommendation — DECIDED

**Option D (Swift-native TF-IDF + exported weights)** was selected after the feasibility spike.

Spike findings:
- Option A blocked: `coremltools` sklearn converter requires sklearn ≤1.5.1; we have 1.8.0
- Option D validated: a simplified 5k-feature LogisticRegression model loses <1% smishing F1 vs the full pipeline (0.945 vs 0.953), weighs only 489 KB as JSON, and inference is ~100 lines of Swift

Implementation is complete in `ios/SMSShield/`. See `ios/README.md` for Xcode setup instructions.

## Development Plan

### Phase 1: Feasibility Spike (1–2 days)

- [ ] Test `coremltools` conversion of our best model
- [ ] Measure converted model size and memory footprint
- [ ] If too large, test with reduced `max_features` (e.g., 5000)
- [ ] If conversion fails, prototype Create ML text classifier on our data
- [ ] Determine which model approach to use going forward

### Phase 2: Extension Skeleton (2–3 days)

- [ ] Create Xcode project with two targets (app + message filter extension)
- [ ] Set up App Group for shared storage
- [ ] Implement `ILMessageFilterExtension` subclass
- [ ] Integrate chosen model (Core ML or Create ML)
- [ ] Map model output to iOS filter actions (allow/junk/promotion)
- [ ] Write filtered message metadata to shared storage

### Phase 3: Companion App UI (2–3 days)

- [ ] SwiftUI list view of filtered messages
- [ ] Filter by prediction class (ham/spam/smishing)
- [ ] Detail view with message text, prediction, confidence, indicators
- [ ] Settings screen (enable/disable filtering)
- [ ] Model info display (name, training strategy, F1 score)

### Phase 4: Testing and Polish (1–2 days)

- [ ] Test with real SMS on physical device (simulator doesn't support message filtering)
- [ ] Verify extension memory usage stays within limits
- [ ] Handle edge cases (empty messages, very long messages, non-English text)
- [ ] App icon and launch screen

## Requirements

- **Xcode 15+** (for ILMessageFilterExtension)
- **iOS 16+** target (for subcategory support: .transaction, .promotion)
- **Physical device** for testing (message filter extensions don't work in simulator)
- **Apple Developer account** (for provisioning profiles with App Group entitlement)

## Risks

| Risk | Mitigation |
|------|------------|
| Model too large for extension memory | Reduce features, use Create ML, or use heuristic fallback |
| Core ML conversion fails | Fall back to Create ML or heuristic |
| Apple rejects app in review | Ensure no message content exfiltration; provide clear privacy policy |
| Low accuracy on real-world messages | Our model was evaluated on real SMS data; monitor and retrain if needed |
| iOS API changes | Target stable APIs (ILMessageFilterExtension has been stable since iOS 11) |

## Privacy Considerations

- All classification happens **on-device** — no message content leaves the phone
- Filtered message log is stored **locally** in the App Group container
- No analytics, no telemetry, no server communication
- This should satisfy both Apple review and user expectations
