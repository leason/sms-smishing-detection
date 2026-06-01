import Foundation

/// Detects heuristic indicators in SMS messages: URLs, emails, phone numbers, suspicious keywords.
struct IndicatorDetector {

    struct Indicators {
        let hasURL: Bool
        let hasEmail: Bool
        let hasPhone: Bool
        let suspiciousKeywords: [String]
    }

    private static let urlPattern = try! NSRegularExpression(pattern: #"https?://|www\."#, options: .caseInsensitive)
    private static let emailPattern = try! NSRegularExpression(pattern: #"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"#)
    private static let phonePattern = try! NSRegularExpression(pattern: #"(?:\+?\d[\d\-\s()]{7,}\d)"#)

    private static let suspiciousKeywords = [
        "urgent", "verify", "account", "locked", "suspended", "click", "login",
        "password", "bank", "refund", "delivery", "package", "prize", "winner",
        "claim", "payment", "tax", "security", "limited time",
    ]

    static func detect(in text: String) -> Indicators {
        let range = NSRange(text.startIndex..., in: text)
        let lower = text.lowercased()

        let hasURL = urlPattern.firstMatch(in: text, range: range) != nil
        let hasEmail = emailPattern.firstMatch(in: text, range: range) != nil
        let hasPhone = phonePattern.firstMatch(in: text, range: range) != nil

        let keywords = suspiciousKeywords.filter { lower.contains($0) }

        return Indicators(
            hasURL: hasURL,
            hasEmail: hasEmail,
            hasPhone: hasPhone,
            suspiciousKeywords: keywords
        )
    }
}
