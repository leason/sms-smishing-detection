import SwiftUI

struct SettingsView: View {
    private let store = MessageStore()
    @State private var messageCount = 0
    @State private var showClearConfirmation = false

    var body: some View {
        NavigationStack {
            List {
                Section("About") {
                    LabeledContent("App", value: "SMS Shield")
                    LabeledContent("Model", value: "TF-IDF + Logistic Regression")
                    LabeledContent("Training Data", value: "Combined (manual + synthetic)")
                    LabeledContent("Vocabulary", value: "5,000 features")
                    LabeledContent("Classes", value: "ham, spam, smishing")
                    LabeledContent("Smishing F1", value: "0.945")
                }

                Section("How It Works") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("SMS Shield filters messages from unknown senders using a machine learning model that runs entirely on your device.")
                        Text("Messages are classified as:")
                            .padding(.top, 4)
                        Label("Ham — safe, normal messages", systemImage: "checkmark.circle")
                            .foregroundColor(.green)
                        Label("Spam — unwanted promotional messages", systemImage: "megaphone")
                            .foregroundColor(.orange)
                        Label("Smishing — phishing attempts via SMS", systemImage: "xmark.shield")
                            .foregroundColor(.red)
                    }
                    .font(.subheadline)
                }

                Section("Setup") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("To enable SMS filtering:")
                            .font(.subheadline.bold())
                        Text("1. Open Settings")
                        Text("2. Go to Messages → Unknown & Spam")
                        Text("3. Enable SMS Shield")
                    }
                    .font(.subheadline)
                }

                Section("Privacy") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("All classification happens on-device. No message content is ever sent to a server.")
                        Text("Filtered message logs are stored locally and never leave your device.")
                    }
                    .font(.subheadline)
                }

                Section("Data") {
                    LabeledContent("Filtered Messages", value: "\(messageCount)")
                    Button("Clear Message Log", role: .destructive) {
                        showClearConfirmation = true
                    }
                }
            }
            .navigationTitle("About")
            .onAppear {
                messageCount = store.messageCount
            }
            .confirmationDialog("Clear all filtered messages?", isPresented: $showClearConfirmation) {
                Button("Clear All", role: .destructive) {
                    store.clearMessages()
                    messageCount = 0
                }
            }
        }
    }
}
