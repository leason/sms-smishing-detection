import SwiftUI

struct MessageDetailView: View {
    let message: FilteredMessage

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Prediction header
                HStack {
                    PredictionBadge(prediction: message.prediction)
                    Text(String(format: "%.1f%% confidence", message.confidence * 100))
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                // Filter action
                HStack {
                    Image(systemName: filterActionIcon)
                        .foregroundColor(filterActionColor)
                    Text("Filter action: \(message.filterAction)")
                        .font(.subheadline)
                }

                Divider()

                // Message text
                VStack(alignment: .leading, spacing: 8) {
                    Text("Message")
                        .font(.headline)
                    Text(message.text)
                        .font(.body)
                        .padding()
                        .background(Color(.systemGray6))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                }

                // Sender and time
                VStack(alignment: .leading, spacing: 4) {
                    Label(message.sender, systemImage: "person.circle")
                    Label(message.receivedAt.formatted(date: .abbreviated, time: .shortened), systemImage: "clock")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)

                Divider()

                // Indicators
                VStack(alignment: .leading, spacing: 8) {
                    Text("Indicators")
                        .font(.headline)
                    IndicatorRow(label: "URL detected", active: message.hasURL, icon: "link")
                    IndicatorRow(label: "Email detected", active: message.hasEmail, icon: "envelope")
                    IndicatorRow(label: "Phone detected", active: message.hasPhone, icon: "phone")
                }

                if !message.suspiciousKeywords.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Suspicious Keywords")
                            .font(.headline)
                        FlowLayout(spacing: 6) {
                            ForEach(message.suspiciousKeywords, id: \.self) { keyword in
                                Text(keyword)
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.red.opacity(0.1))
                                    .foregroundColor(.red)
                                    .clipShape(Capsule())
                            }
                        }
                    }
                }

                Text("Indicators are heuristic signals, not proof of malicious intent.")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .italic()
            }
            .padding()
        }
        .navigationTitle("Message Detail")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var filterActionIcon: String {
        switch message.filterAction {
        case "junk": return "xmark.shield.fill"
        case "promotion": return "megaphone.fill"
        default: return "checkmark.shield.fill"
        }
    }

    private var filterActionColor: Color {
        switch message.filterAction {
        case "junk": return .red
        case "promotion": return .orange
        default: return .green
        }
    }
}

struct IndicatorRow: View {
    let label: String
    let active: Bool
    let icon: String

    var body: some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(active ? .red : .secondary)
            Text(label)
            Spacer()
            Image(systemName: active ? "checkmark.circle.fill" : "minus.circle")
                .foregroundColor(active ? .red : .secondary)
        }
        .font(.subheadline)
    }
}

/// Simple flow layout for keyword chips
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = arrange(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y),
                                  proposal: .unspecified)
        }
    }

    private func arrange(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (CGSize(width: maxWidth, height: y + rowHeight), positions)
    }
}
