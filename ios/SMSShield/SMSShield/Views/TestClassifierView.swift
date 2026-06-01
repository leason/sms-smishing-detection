import SwiftUI

/// Allows testing the classifier locally by typing or pasting SMS messages.
struct TestClassifierView: View {
    @State private var inputText = ""
    @State private var prediction: SMSClassifier.Prediction?
    @State private var indicators: IndicatorDetector.Indicators?
    @State private var classifier: SMSClassifier?
    @State private var loadError: String?

    private let sampleMessages = [
        ("Ham", "Hey! Are we still on for lunch tomorrow? Let me know."),
        ("Spam", "CONGRATULATIONS! You've been selected for a FREE cruise vacation! Call now!"),
        ("Smishing", "ALERT: Your bank account has been locked. Verify your identity now at http://secure-bank-login.com/verify"),
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Input
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Enter SMS Text")
                            .font(.headline)
                        TextEditor(text: $inputText)
                            .frame(minHeight: 100)
                            .padding(4)
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.3)))

                        Button(action: classify) {
                            Label("Classify", systemImage: "text.magnifyingglass")
                                .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }

                    // Sample messages
                    DisclosureGroup("Try a sample") {
                        ForEach(sampleMessages, id: \.0) { label, text in
                            Button {
                                inputText = text
                                classify()
                            } label: {
                                VStack(alignment: .leading) {
                                    Text(label).font(.caption.bold())
                                    Text(text).font(.caption).lineLimit(2)
                                }
                            }
                            .buttonStyle(.plain)
                            .padding(.vertical, 4)
                        }
                    }

                    // Results
                    if let pred = prediction {
                        Divider()
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                PredictionBadge(prediction: pred.label)
                                Text(String(format: "%.1f%% confidence", pred.confidence * 100))
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }

                            // Probability bars
                            ForEach(["ham", "spam", "smishing"], id: \.self) { cls in
                                let prob = pred.probabilities[cls] ?? 0
                                HStack {
                                    Text(cls)
                                        .font(.caption)
                                        .frame(width: 60, alignment: .trailing)
                                    GeometryReader { geo in
                                        RoundedRectangle(cornerRadius: 3)
                                            .fill(barColor(cls))
                                            .frame(width: geo.size.width * prob)
                                    }
                                    .frame(height: 12)
                                    Text(String(format: "%.1f%%", prob * 100))
                                        .font(.caption)
                                        .frame(width: 50, alignment: .leading)
                                }
                            }

                            if let ind = indicators {
                                Divider()
                                HStack(spacing: 12) {
                                    IndicatorChip(label: "URL", active: ind.hasURL)
                                    IndicatorChip(label: "Email", active: ind.hasEmail)
                                    IndicatorChip(label: "Phone", active: ind.hasPhone)
                                }
                                if !ind.suspiciousKeywords.isEmpty {
                                    Text("Keywords: \(ind.suspiciousKeywords.joined(separator: ", "))")
                                        .font(.caption)
                                        .foregroundColor(.red)
                                }
                            }
                        }
                    }

                    if let error = loadError {
                        Text("Model error: \(error)")
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
                .padding()
            }
            .navigationTitle("Test Classifier")
            .onAppear(perform: loadModel)
        }
    }

    private func loadModel() {
        guard classifier == nil else { return }
        guard let url = Bundle.main.url(forResource: "SMSFilterModel", withExtension: "json") else {
            loadError = "Model file not found in bundle"
            return
        }
        do {
            classifier = try SMSClassifier(modelURL: url)
        } catch {
            loadError = error.localizedDescription
        }
    }

    private func classify() {
        guard let clf = classifier else { return }
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        prediction = clf.predict(text)
        indicators = IndicatorDetector.detect(in: text)
    }

    private func barColor(_ cls: String) -> Color {
        switch cls {
        case "ham": return .green
        case "spam": return .orange
        case "smishing": return .red
        default: return .gray
        }
    }
}

struct IndicatorChip: View {
    let label: String
    let active: Bool

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: active ? "checkmark.circle.fill" : "minus.circle")
                .font(.caption2)
            Text(label)
                .font(.caption2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(active ? Color.red.opacity(0.1) : Color.secondary.opacity(0.1))
        .foregroundColor(active ? .red : .secondary)
        .clipShape(Capsule())
    }
}
