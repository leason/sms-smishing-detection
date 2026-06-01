import Foundation

/// A filtered SMS message stored in the shared App Group container.
struct FilteredMessage: Codable, Identifiable {
    let id: UUID
    let text: String
    let sender: String
    let receivedAt: Date
    let prediction: String        // "ham", "spam", "smishing"
    let confidence: Double
    let filterAction: String      // "allow", "junk", "promotion"
    let hasURL: Bool
    let hasEmail: Bool
    let hasPhone: Bool
    let suspiciousKeywords: [String]

    init(
        text: String,
        sender: String,
        prediction: String,
        confidence: Double,
        filterAction: String,
        hasURL: Bool = false,
        hasEmail: Bool = false,
        hasPhone: Bool = false,
        suspiciousKeywords: [String] = []
    ) {
        self.id = UUID()
        self.text = text
        self.sender = sender
        self.receivedAt = Date()
        self.prediction = prediction
        self.confidence = confidence
        self.filterAction = filterAction
        self.hasURL = hasURL
        self.hasEmail = hasEmail
        self.hasPhone = hasPhone
        self.suspiciousKeywords = suspiciousKeywords
    }
}
