# SMS Shield — iOS SMS Filtering App

An iOS app that uses the smishing detection model from this project to filter incoming SMS messages from unknown senders.

## Architecture

- **Message Filter Extension**: Classifies incoming SMS using a lightweight TF-IDF + Logistic Regression model running entirely on-device
- **Companion App**: SwiftUI interface for browsing filtered messages, testing the classifier, and viewing model info
- **Shared Storage**: App Group container with JSON-based message log shared between extension and app

## Model

The iOS model is a simplified version of the main project's best model:

| Property | Value |
|----------|-------|
| Algorithm | TF-IDF + Logistic Regression |
| Vocabulary | 5,000 features (unigrams + bigrams) |
| Model file | `SMSFilterModel.json` (489 KB) |
| Smishing F1 | 0.945 (vs 0.953 for full pipeline) |
| Macro F1 | 0.952 (vs 0.963 for full pipeline) |

The model weights (vocabulary, IDF values, logistic regression coefficients) are exported as JSON and the TF-IDF + softmax inference is implemented natively in Swift (~100 lines).

## Project Structure

```
ios/
├── README.md
├── SMSShield/
│   ├── Shared/                          # Shared between app and extension
│   │   ├── SMSClassifier.swift          # TF-IDF + LogReg inference engine
│   │   ├── SMSFilterModel.json          # Exported model weights (489 KB)
│   │   ├── FilteredMessage.swift        # Data model for filtered messages
│   │   ├── MessageStore.swift           # App Group JSON persistence
│   │   └── IndicatorDetector.swift      # URL/email/phone/keyword detection
│   ├── MessageFilter/                   # Message Filter Extension target
│   │   ├── MessageFilterExtension.swift # ILMessageFilterExtension impl
│   │   └── Info.plist                   # Extension configuration
│   └── SMSShield/                       # Companion App target
│       ├── SMSShieldApp.swift           # App entry point
│       └── Views/
│           ├── ContentView.swift        # Tab navigation
│           ├── MessageListView.swift    # Filtered message list
│           ├── MessageDetailView.swift  # Message detail with indicators
│           ├── TestClassifierView.swift # Interactive classifier testing
│           └── SettingsView.swift       # About, setup instructions, privacy
```

## Xcode Project Setup

The Swift source files are provided but the Xcode project must be created manually (requires Xcode GUI for extension targets and entitlements):

### 1. Create Project

1. Open Xcode → File → New → Project
2. Choose **App** template, name it **SMSShield**
3. Set Team, Bundle ID (e.g., `com.leason.smsshield`)
4. Language: Swift, Interface: SwiftUI
5. Save in the `ios/` directory

### 2. Add Message Filter Extension

1. File → New → Target
2. Choose **Message Filter Extension**
3. Name it **MessageFilter**
4. Activate the scheme when prompted

### 3. Configure App Groups

1. Select the **SMSShield** app target → Signing & Capabilities
2. Add **App Groups** capability
3. Create group: `group.com.leason.smsshield`
4. Select the **MessageFilter** extension target → Signing & Capabilities
5. Add the same App Group

### 4. Add Source Files

1. Drag the `Shared/` folder into both targets (check both SMSShield and MessageFilter)
2. Drag `MessageFilter/MessageFilterExtension.swift` into the MessageFilter target
3. Drag `SMSShield/` app files into the SMSShield target
4. Add `SMSFilterModel.json` to both targets' "Copy Bundle Resources" build phase

### 5. Build and Test

- Build for a **physical iOS device** (message filtering doesn't work in simulator)
- Go to Settings → Messages → Unknown & Spam → enable SMS Shield
- Test with the built-in classifier test tab first

## Regenerating Model Weights

If the Python model is retrained, regenerate the iOS weights:

```bash
python3 ios/export_model.py
```

This trains the simplified 5k-feature model and exports `SMSFilterModel.json`.

## Limitations

- Only filters messages from **unknown senders** (iOS limitation)
- ~0.8% lower smishing F1 than the full Python pipeline (5k vs 36k features, no indicator features in model)
- Extension memory limit (~6 MB) constrains vocabulary size
- Cannot test message filtering in iOS Simulator — requires physical device
- Model is static; does not learn from user corrections
