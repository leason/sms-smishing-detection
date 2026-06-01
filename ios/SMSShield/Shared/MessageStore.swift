import Foundation

/// Persists filtered messages to the shared App Group container using JSON.
/// Both the Message Filter Extension and the companion app use this store.
final class MessageStore {

    // IMPORTANT: Replace with your actual App Group identifier after creating it
    // in your Apple Developer portal and Xcode capabilities.
    static let appGroupID = "group.com.leason.smsshield"

    private let fileURL: URL

    init() {
        let containerURL = FileManager.default
            .containerURL(forSecurityApplicationGroupIdentifier: Self.appGroupID)
            ?? FileManager.default.temporaryDirectory
        self.fileURL = containerURL.appendingPathComponent("filtered_messages.json")
    }

    func loadMessages() -> [FilteredMessage] {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return [] }
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            return try decoder.decode([FilteredMessage].self, from: data)
        } catch {
            print("MessageStore: failed to load messages: \(error)")
            return []
        }
    }

    func saveMessage(_ message: FilteredMessage) {
        var messages = loadMessages()
        messages.insert(message, at: 0)

        // Keep only the last 500 messages to limit storage
        if messages.count > 500 {
            messages = Array(messages.prefix(500))
        }

        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(messages)
            try data.write(to: fileURL, options: .atomic)
        } catch {
            print("MessageStore: failed to save message: \(error)")
        }
    }

    func clearMessages() {
        try? FileManager.default.removeItem(at: fileURL)
    }

    var messageCount: Int {
        loadMessages().count
    }
}
