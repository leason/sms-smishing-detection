import IdentityLookup

/// iOS Message Filter Extension that classifies incoming SMS from unknown senders.
/// Uses a lightweight TF-IDF + Logistic Regression model running entirely on-device.
final class MessageFilterExtension: ILMessageFilterExtension {}

extension MessageFilterExtension: ILMessageFilterQueryHandling {

    func handle(
        _ queryRequest: ILMessageFilterQueryRequest,
        context: ILMessageFilterExtensionContext,
        completion: @escaping (ILMessageFilterQueryResponse) -> Void
    ) {
        let response = ILMessageFilterQueryResponse()

        guard let messageBody = queryRequest.messageBody, !messageBody.isEmpty else {
            response.action = .allow
            completion(response)
            return
        }

        let sender = queryRequest.sender ?? "Unknown"

        // Classify the message
        do {
            let classifier = try loadClassifier()
            let prediction = classifier.predict(messageBody)
            let indicators = IndicatorDetector.detect(in: messageBody)

            // Map prediction to iOS filter action
            let filterAction: ILMessageFilterAction
            switch prediction.label {
            case "smishing":
                filterAction = .junk
            case "spam":
                filterAction = .promotion
            default:
                filterAction = .allow
            }

            response.action = filterAction

            // Log to shared storage
            let store = MessageStore()
            let filteredMessage = FilteredMessage(
                text: messageBody,
                sender: sender,
                prediction: prediction.label,
                confidence: prediction.confidence,
                filterAction: filterActionString(filterAction),
                hasURL: indicators.hasURL,
                hasEmail: indicators.hasEmail,
                hasPhone: indicators.hasPhone,
                suspiciousKeywords: indicators.suspiciousKeywords
            )
            store.saveMessage(filteredMessage)

        } catch {
            // If model fails to load, allow the message through
            response.action = .allow
        }

        completion(response)
    }

    private func loadClassifier() throws -> SMSClassifier {
        guard let modelURL = Bundle(for: type(of: self))
            .url(forResource: "SMSFilterModel", withExtension: "json") else {
            throw NSError(domain: "MessageFilter", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Model file not found"])
        }
        return try SMSClassifier(modelURL: modelURL)
    }

    private func filterActionString(_ action: ILMessageFilterAction) -> String {
        switch action {
        case .allow: return "allow"
        case .junk: return "junk"
        case .promotion: return "promotion"
        default: return "none"
        }
    }
}
