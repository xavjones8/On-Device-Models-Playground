/*
 ChatView.swift
 
 General-purpose chat interface using Apple Foundation Models.
 
 Features:
 - Multi-turn conversation with context
 - Streaming responses (real-time token display)
 - Markdown rendering for AI responses
 - Copy functionality (button + long-press)
 - Availability checking for Apple Intelligence
 
 Architecture:
 - ChatViewModel: Manages conversation state and AI session
 - ChatView: SwiftUI view with message list and input
 - MessageBubble: Individual message display with styling
 
 Apple Intelligence Requirements:
 - iPhone 15 Pro or later / M-series Mac
 - iOS 18.1+ / macOS 15.1+
 - Apple Intelligence enabled in Settings
 */

import SwiftUI
import FoundationModels
import Combine
import MarkdownUI

struct ChatMessage: Identifiable, Equatable {
    let id: UUID
    var content: String
    let isUser: Bool
    var isStreaming: Bool
    
    init(id: UUID = UUID(), content: String, isUser: Bool, isStreaming: Bool = false) {
        self.id = id
        self.content = content
        self.isUser = isUser
        self.isStreaming = isStreaming
    }
}

@MainActor
class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessage] = []
    @Published var inputText: String = ""
    @Published var isGenerating: Bool = false
    @Published var availability: FoundationModelAvailability = .unknown
    @Published var errorMessage: String?
    
    private var session: LanguageModelSession?
    private let maxMessagesToKeep = 6 // Keep last 3 exchanges when resetting
    
    private let systemInstructions = """
        You are a helpful and friendly AI assistant. 
        Respond naturally and conversationally to the user's questions and requests.
        Be concise but thorough, and maintain a warm, approachable tone.
        """
    
    init() {
        checkAvailability()
        setupSession()
    }
    
    private func setupSession() {
        guard availability == .available else { return }
        session = LanguageModelSession(instructions: systemInstructions)
    }
    
    func clearConversation() {
        messages = []
        errorMessage = nil
        session = nil  // Explicitly clear old session
        setupSession()
        print("🟢 Session cleared and recreated")
    }
    
    /// Reset session but preserve recent messages for context continuity
    private func resetSessionWithContext() {
        // Keep only the most recent messages
        let recentMessages = Array(messages.suffix(maxMessagesToKeep))
        
        // Create new session
        setupSession()
        
        // Rebuild context by re-adding recent messages to the new session
        // Note: We keep UI messages but the session starts fresh
        // The AI won't have full history but user sees continuity
        messages = recentMessages
        
        // Add a system note about context reset
        let systemNote = ChatMessage(
            content: "_(Conversation trimmed to stay within limits)_",
            isUser: false,
            isStreaming: false
        )
        messages.insert(systemNote, at: 0)
    }
    
    func checkAvailability() {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            availability = .available
        case .unavailable(.deviceNotEligible):
            availability = .notEligible
        case .unavailable(.appleIntelligenceNotEnabled):
            availability = .notEnabled
        case .unavailable(.modelNotReady):
            availability = .notReady
        default:
            availability = .unknown
        }
    }
    
    func sendMessage() {
        guard !inputText.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let prompt = inputText
        inputText = ""
        executePrompt(prompt, isRetry: false)
    }
    
    private func executePrompt(_ prompt: String, isRetry: Bool) {
        guard availability == .available else {
            errorMessage = "Apple Intelligence is not available"
            return
        }
        
        guard let currentSession = session else {
            errorMessage = "AI session failed to initialize"
            return
        }
        
        // Add user message (only if not a retry)
        if !isRetry {
            let userMessage = ChatMessage(content: prompt, isUser: true, isStreaming: false)
            messages.append(userMessage)
        }
        
        isGenerating = true
        errorMessage = nil
        
        // Add placeholder for AI response
        let aiMessageId = UUID()
        let streamingMessage = ChatMessage(id: aiMessageId, content: "", isUser: false, isStreaming: true)
        messages.append(streamingMessage)
        
        Task {
            do {
                let options = GenerationOptions(temperature: 0.7)
                
                let stream = currentSession.streamResponse(
                    to: prompt,
                    options: options
                )
                
                for try await partialResponse in stream {
                    if let index = messages.firstIndex(where: { $0.id == aiMessageId }) {
                        messages[index].content = partialResponse.content
                    }
                }
                
                // Mark as complete
                if let index = messages.firstIndex(where: { $0.id == aiMessageId }) {
                    messages[index].isStreaming = false
                }
                isGenerating = false
                
            } catch {
                let errorString = "\(error)"
                let errorDescribed = String(describing: error)
                let errorLocalized = error.localizedDescription
                let allErrorText = "\(errorString) \(errorDescribed) \(errorLocalized)".lowercased()
                
                let isContextOverflow = allErrorText.contains("context length") || 
                                        allErrorText.contains("4096") || 
                                        allErrorText.contains("exceeded") ||
                                        allErrorText.contains("inferenceerror") ||
                                        allErrorText.contains("inferencefailed") ||
                                        allErrorText.contains("modelmanagererror") ||
                                        allErrorText.contains("code=1001") ||
                                        allErrorText.contains("code=-1")
                
                print("🔴 Chat error: \(errorString)")
                print("🔴 Context overflow detected: \(isContextOverflow), isRetry: \(isRetry)")
                
                // Remove failed AI message
                messages.removeAll { $0.id == aiMessageId }
                
                if isContextOverflow && !isRetry {
                    // Remove the user message we just added
                    if let userMsg = messages.last, userMsg.isUser {
                        messages.removeLast()
                    }
                    
                    // Full reset - clear and start fresh
                    clearConversation()
                    
                    errorMessage = "Context limit reached. Retrying with fresh session..."
                    
                    // Small delay then retry
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    errorMessage = nil
                    
                    // Retry with fresh session
                    executePrompt(prompt, isRetry: true)
                    return
                    
                } else if isContextOverflow && isRetry {
                    errorMessage = "Request too large. Try a shorter message."
                    isGenerating = false
                } else {
                    errorMessage = "Failed: \(error.localizedDescription)"
                    isGenerating = false
                }
            }
        }
    }
}

struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    @FocusState private var isInputFocused: Bool
    
    var body: some View {
        NavigationStack {
            ZStack {
                Color(UIColor.systemBackground)
                    .ignoresSafeArea()
                
                VStack(spacing: 0) {
                    // Messages
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(spacing: 16) {
                                // Availability message if needed
                                if viewModel.availability != .available {
                                    AvailabilityBanner(availability: viewModel.availability)
                                        .padding(.horizontal)
                                        .padding(.top, 8)
                                }
                                
                                // Error message
                                if let error = viewModel.errorMessage {
                                    Text(error)
                                        .font(.caption)
                                        .foregroundColor(.red)
                                        .padding(.horizontal)
                                }
                                
                                // Chat messages
                                ForEach(viewModel.messages) { message in
                                    MessageBubble(message: message)
                                        .id(message.id)
                                }
                            }
                            .padding(.vertical)
                        }
                        .onChange(of: viewModel.messages.count) { _ in
                            if let lastMessage = viewModel.messages.last {
                                withAnimation(.easeOut(duration: 0.3)) {
                                    proxy.scrollTo(lastMessage.id, anchor: .bottom)
                                }
                            }
                        }
                        .onChange(of: viewModel.messages.last?.content ?? "") { _ in
                            if let lastMessage = viewModel.messages.last, lastMessage.isStreaming {
                                withAnimation(.easeOut(duration: 0.1)) {
                                    proxy.scrollTo(lastMessage.id, anchor: .bottom)
                                }
                            }
                        }
                    }
                    
                    // Input area
                    VStack(spacing: 0) {
                        Divider()
                        
                        HStack(spacing: 12) {
                            TextField("Type a message...", text: $viewModel.inputText, axis: .vertical)
                                .textFieldStyle(.plain)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 12)
                                .background(Color(UIColor.secondarySystemBackground))
                                .cornerRadius(24)
                                .lineLimit(1...5)
                                .focused($isInputFocused)
                                .disabled(viewModel.isGenerating || viewModel.availability != .available)
                            
                            Button(action: {
                                viewModel.sendMessage()
                                isInputFocused = false
                            }) {
                                Image(systemName: viewModel.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                                    .font(.system(size: 32))
                                    .foregroundColor(
                                        viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty || 
                                        viewModel.isGenerating || 
                                        viewModel.availability != .available
                                        ? .gray : .blue
                                    )
                            }
                            .disabled(
                                viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty || 
                                viewModel.isGenerating || 
                                viewModel.availability != .available
                            )
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                        .background(Color(UIColor.systemBackground))
                    }
                }
            }
            .navigationTitle("AI Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.messages.isEmpty {
                        Button("New Chat") {
                            viewModel.clearConversation()
                        }
                        .foregroundColor(.blue)
                    }
                }
            }
        }
    }
}

struct MessageBubble: View {
    let message: ChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer(minLength: 60)
            }
            
            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                ZStack(alignment: .topTrailing) {
                    if message.isUser {
                        // User messages: plain text
                        Text(message.content)
                            .font(.body)
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .background(Color.blue)
                            .cornerRadius(20)
                    } else {
                        // AI messages: markdown
                        Markdown(message.content)
                            .markdownTheme(.basic)
                            .markdownTextStyle(\.code) {
                                FontFamilyVariant(.monospaced)
                                FontSize(.em(0.85))
                                ForegroundColor(.primary)
                                BackgroundColor(.clear)
                            }
                            .markdownBlockStyle(\.codeBlock) { configuration in
                                ScrollView(.horizontal, showsIndicators: false) {
                                    configuration.label
                                        .background(.clear)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                }
                                 
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 12)
                            .cornerRadius(20)
                    }
                    
                    // Copy button for AI messages
                    if !message.isUser && !message.content.isEmpty {
                        Button(action: copyMessage) {
                            Image(systemName: "doc.on.doc")
                                .font(.caption)
                                .foregroundColor(.secondary)
                                .padding(6)
                                .background(Color(UIColor.tertiarySystemBackground))
                                .clipShape(Circle())
                        }
                        .opacity(0.7)
                        .offset(x: 4, y: -4)
                    }
                }
                .onLongPressGesture {
                    if !message.isUser {
                        copyMessage()
                    }
                }
                
                if message.isStreaming {
                    HStack(spacing: 4) {
                        ForEach(0..<3) { index in
                            Circle()
                                .fill(Color.secondary.opacity(0.5))
                                .frame(width: 6, height: 6)
                                .offset(y: 0)
                                .animation(
                                    Animation.easeInOut(duration: 0.6)
                                        .repeatForever()
                                        .delay(Double(index) * 0.2),
                                    value: message.isStreaming
                                )
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.top, 4)
                }
            }
            
            if !message.isUser {
                Spacer(minLength: 60)
            }
        }
        .padding(.horizontal)
    }
    
    private func copyMessage() {
        UIPasteboard.general.string = message.content
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
    }
}

struct AvailabilityBanner: View {
    let availability: FoundationModelAvailability
    
    var body: some View {
        HStack {
            Image(systemName: iconName)
                .foregroundColor(.orange)
            Text(message)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.1))
        .cornerRadius(12)
    }
    
    var iconName: String {
        switch availability {
        case .notEligible:
            return "iphone.slash"
        case .notEnabled:
            return "gear"
        case .notReady:
            return "arrow.down.circle"
        default:
            return "exclamationmark.triangle"
        }
    }
    
    var message: String {
        switch availability {
        case .notEligible:
            return "This device doesn't support Apple Intelligence"
        case .notEnabled:
            return "Enable Apple Intelligence in Settings"
        case .notReady:
            return "Apple Intelligence is downloading..."
        default:
            return "Apple Intelligence is unavailable"
        }
    }
}

#Preview {
    ChatView()
}

