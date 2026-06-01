import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            MessageListView()
                .tabItem {
                    Label("Messages", systemImage: "message.fill")
                }
                .tag(0)

            TestClassifierView()
                .tabItem {
                    Label("Test", systemImage: "text.magnifyingglass")
                }
                .tag(1)

            SettingsView()
                .tabItem {
                    Label("About", systemImage: "info.circle")
                }
                .tag(2)
        }
    }
}
