import SwiftUI

struct MessageListView: View {
    @State private var messages: [FilteredMessage] = []
    @State private var filterType: String = "all"

    private let store = MessageStore()

    private var filteredMessages: [FilteredMessage] {
        if filterType == "all" { return messages }
        return messages.filter { $0.prediction == filterType }
    }

    var body: some View {
        NavigationStack {
            Group {
                if messages.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "message.badge.checkmark")
                            .font(.system(size: 48))
                            .foregroundColor(.secondary)
                        Text("No Filtered Messages")
                            .font(.headline)
                        Text("Messages from unknown senders will appear here after they are filtered.")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                } else {
                    List(filteredMessages) { message in
                        NavigationLink(destination: MessageDetailView(message: message)) {
                            MessageRow(message: message)
                        }
                    }
                }
            }
            .navigationTitle("Filtered Messages")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button("All") { filterType = "all" }
                        Button("Ham") { filterType = "ham" }
                        Button("Spam") { filterType = "spam" }
                        Button("Smishing") { filterType = "smishing" }
                    } label: {
                        Label("Filter", systemImage: "line.3.horizontal.decrease.circle")
                    }
                }
            }
            .onAppear {
                messages = store.loadMessages()
            }
            .refreshable {
                messages = store.loadMessages()
            }
        }
    }
}

struct MessageRow: View {
    let message: FilteredMessage

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                PredictionBadge(prediction: message.prediction)
                Spacer()
                Text(message.receivedAt, style: .relative)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Text(message.text)
                .font(.subheadline)
                .lineLimit(2)
                .foregroundColor(.primary)
            Text("From: \(message.sender)")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

struct PredictionBadge: View {
    let prediction: String

    private var color: Color {
        switch prediction {
        case "ham": return .green
        case "spam": return .orange
        case "smishing": return .red
        default: return .gray
        }
    }

    var body: some View {
        Text(prediction.uppercased())
            .font(.caption2.bold())
            .padding(.horizontal, 8)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .clipShape(Capsule())
    }
}
