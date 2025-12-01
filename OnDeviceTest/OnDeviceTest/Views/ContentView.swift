/*
 ContentView.swift
 
 Main view containing the tab-based navigation.
 
 Tabs:
 1. Analyzer - Prompt classification and AI response generation
 2. Chat - General conversation with Apple Intelligence
 3. Research - Stock analysis with tool-calling
 4. Debug - API testing and debugging tools
 
 This file also contains PromptAnalyzerView which handles:
 - Text input for prompts
 - CoreML classification display (task type, complexity scores)
 - AI-generated responses using Foundation Models
 - Collapsible result cards
 */

import SwiftUI
import MarkdownUI

struct ContentView: View {
    var body: some View {
        TabView {
            PromptAnalyzerView()
                .tabItem {
                    Label("Analyzer", systemImage: "chart.bar.doc.horizontal")
                }
            
            ChatView()
                .tabItem {
                    Label("Chat", systemImage: "message.fill")
                }
            
            StockResearchView()
                .tabItem {
                    Label("Research", systemImage: "chart.line.uptrend.xyaxis")
                }
            
            APIDebugView()
                .tabItem {
                    Label("Debug", systemImage: "ladybug.fill")
                }
        }
    }
}

// MARK: - Prompt Analyzer View

struct PromptAnalyzerView: View {
    @StateObject private var router = PromptRouter()
    @State private var promptText: String = ""
    @FocusState private var isInputFocused: Bool
    
    var isProcessing: Bool {
        router.isGrading || router.foundationIsGenerating
    }
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 24) {
                    // Model Loading Indicator
                    if router.isModelLoading {
                        VStack(spacing: 12) {
                            ProgressView()
                                .scaleEffect(1.2)
                            Text("Loading AI Model...")
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 40)
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(16)
                        .padding(.horizontal)
                    }
                    
                    // Prompt Input Card
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Enter your prompt", systemImage: "text.cursor")
                            .font(.headline)
                            .foregroundColor(.primary.opacity(0.8))
                        
                        TextField("What would you like to analyze?", text: $promptText, axis: .vertical)
                            .textFieldStyle(.plain)
                            .padding(14)
                            .background(Color(UIColor.tertiarySystemBackground))
                            .cornerRadius(12)
                            .lineLimit(3...8)
                            .focused($isInputFocused)
                            .disabled(router.isModelLoading)
                    }
                    .padding(20)
                    .background(Color(UIColor.secondarySystemBackground))
                    .cornerRadius(16)
                    .padding(.horizontal)
                    .opacity(router.isModelLoading ? 0.5 : 1.0)
                    
                    // Analyze Button
                    Button(action: evaluatePrompt) {
                        HStack(spacing: 10) {
                            if isProcessing {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Image(systemName: "wand.and.stars")
                            }
                            Text(buttonText)
                                .fontWeight(.semibold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(
                            LinearGradient(
                                colors: promptText.isEmpty || isProcessing || router.isModelLoading
                                    ? [Color.gray.opacity(0.5), Color.gray.opacity(0.3)]
                                    : [Color.blue, Color.purple],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .foregroundColor(.white)
                        .cornerRadius(14)
                    }
                    .padding(.horizontal)
                    .disabled(promptText.isEmpty || isProcessing || router.isModelLoading)
                    
                    // Error Messages
                    if let error = router.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }
                    
                    if let error = router.foundationErrorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .padding(.horizontal)
                    }
                    
                    // AI Response Card (between button and classifiers)
                    if router.foundationIsGenerating || !router.foundationResponse.isEmpty {
                        AIResponseCard(
                            response: router.foundationResponse,
                            isGenerating: router.foundationIsGenerating,
                            availability: router.foundationAvailability
                        )
                    }
                    
                    // Results (classifiers)
                    if let scores = router.lastScores {
                        VStack(spacing: 20) {
                            // Task Type Card
                            TaskTypeCard(taskType: scores.taskType)
                            
                            // Complexity Scores Card
                            ComplexityCard(scores: scores)
                        }
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: router.lastScores != nil)
                    }
                    
                    Spacer(minLength: 40)
                }
                .padding(.top, 20)
            }
            .background(Color(UIColor.systemBackground))
            .navigationTitle("Prompt Analyzer")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
    
    var buttonText: String {
        if router.isGrading {
            return "Analyzing..."
        } else if router.foundationIsGenerating {
            return "Generating..."
        } else {
            return "Analyze Prompt"
        }
    }
    
    func evaluatePrompt() {
        isInputFocused = false // Dismiss keyboard
        Task {
            await router.analyzeAndRespond(prompt: promptText)
        }
    }
}

// MARK: - Task Type Card

struct TaskTypeCard: View {
    let taskType: TaskTypePrediction
    @State private var isExpanded: Bool = false
    
    var sortedTasks: [(name: String, prob: Double)] {
        taskType.allProbabilities
            .sorted { $0.value > $1.value }
            .map { (name: $0.key, prob: $0.value) }
            .filter { $0.prob >= 0.1 } // Only show categories above 10%
    }
    
    var topTask: String {
        sortedTasks.first?.name ?? ""
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header with collapse button
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Image(systemName: "tag.fill")
                        .foregroundColor(.purple)
                    Text("Task Classification")
                        .font(.headline)
                    Spacer()
                    
                    // Primary result always visible
                    HStack(spacing: 12) {
                        VStack(alignment: .trailing, spacing: 2) {
                            Text(taskType.taskType1)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundColor(.purple)
                            Text("\(Int(taskType.probability * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                // Top Prediction Highlight
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Primary Task")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(taskType.taskType1)
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.purple)
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("Confidence")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text("\(Int(taskType.probability * 100))%")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.purple)
                    }
                }
                .padding()
                .background(Color.purple.opacity(0.1))
                .cornerRadius(12)
                
                // Secondary task if applicable
                if taskType.taskType2 != "NA" {
                    HStack {
                        Text("Also likely:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(taskType.taskType2)
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                }
                
                // All Task Types Chart (filtered to >10%)
                if !sortedTasks.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("All Categories (>10%)")
                            .font(.subheadline)
                            .fontWeight(.medium)
                            .foregroundColor(.secondary)
                        
                        ForEach(sortedTasks, id: \.name) { task in
                            TaskTypeBar(
                                name: task.name,
                                probability: task.prob,
                                isTop: task.name == topTask
                            )
                        }
                    }
                }
            }
        }
        .padding(20)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .padding(.horizontal)
    }
}

struct TaskTypeBar: View {
    let name: String
    let probability: Double
    let isTop: Bool
    
    var barColor: Color {
        if isTop {
            return .purple
        }
        return probability > 0.1 ? .purple.opacity(0.6) : .gray.opacity(0.4)
    }
    
    var body: some View {
        HStack(spacing: 10) {
            Text(name)
                .font(.caption)
                .fontWeight(isTop ? .bold : .regular)
                .foregroundColor(isTop ? .primary : .secondary)
                .frame(width: 90, alignment: .leading)
            
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.15))
                    
                    RoundedRectangle(cornerRadius: 4)
                        .fill(barColor)
                        .frame(width: max(geo.size.width * probability, probability > 0.01 ? 4 : 0))
                }
            }
            .frame(height: isTop ? 16 : 12)
            
            Text(String(format: "%.1f%%", probability * 100))
                .font(.caption2)
                .fontWeight(isTop ? .bold : .regular)
                .foregroundColor(isTop ? .purple : .secondary)
                .monospacedDigit()
                .frame(width: 44, alignment: .trailing)
        }
        .padding(.vertical, isTop ? 4 : 2)
        .padding(.horizontal, isTop ? 8 : 0)
        .background(isTop ? Color.purple.opacity(0.08) : Color.clear)
        .cornerRadius(8)
    }
}

// MARK: - Complexity Card

struct ComplexityCard: View {
    let scores: ComplexityScores
    @State private var isExpanded: Bool = false
    
    var complexityItems: [(label: String, value: Double, icon: String)] {
        [
            ("Creativity", scores.creativity, "paintbrush.fill"),
            ("Reasoning", scores.reasoning, "brain.head.profile"),
            ("Constraints", scores.constraint, "checklist"),
            ("Domain Knowledge", scores.domainKnowledge, "book.fill"),
            ("Context Required", scores.contextualKnowledge, "doc.text.fill"),
            ("Few-Shot Examples", scores.fewShots, "list.number")
        ]
    }
    
    var aggregateColor: Color {
        if scores.aggregate > 0.7 { return .red }
        if scores.aggregate > 0.4 { return .orange }
        return .green
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header with collapse button
            Button(action: {
                withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Image(systemName: "chart.bar.fill")
                        .foregroundColor(.blue)
                    Text("Complexity Analysis")
                        .font(.headline)
                    Spacer()
                    
                    // Primary result always visible
                    HStack(spacing: 12) {
                        VStack(alignment: .trailing, spacing: 2) {
                            Text("Overall")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            Text(String(format: "%.2f", scores.aggregate))
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(aggregateColor)
                        }
                        
                        Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                // Detailed header
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Image(systemName: "chart.bar.fill")
                                .foregroundColor(.blue)
                            Text("Complexity Analysis")
                                .font(.headline)
                        }
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text("Overall")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.2f", scores.aggregate))
                            .font(.title3)
                            .fontWeight(.bold)
                            .foregroundColor(aggregateColor)
                    }
                }
                
                // Score Bars
                ForEach(complexityItems, id: \.label) { item in
                    ComplexityScoreRow(
                        label: item.label,
                        score: item.value,
                        icon: item.icon
                    )
                }
            }
        }
        .padding(20)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .padding(.horizontal)
    }
}

struct ComplexityScoreRow: View {
    let label: String
    let score: Double
    let icon: String
    
    var scoreColor: Color {
        if score > 0.7 { return .red }
        if score > 0.5 { return .orange }
        if score > 0.3 { return .yellow }
        return .green
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(width: 16)
                Text(label)
                    .font(.subheadline)
                    .foregroundColor(.primary.opacity(0.9))
                Spacer()
                Text(String(format: "%.3f", score))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .monospacedDigit()
                    .foregroundColor(scoreColor)
            }
            
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 4)
                        .fill(Color.gray.opacity(0.2))
                    
                    RoundedRectangle(cornerRadius: 4)
                        .fill(
                            LinearGradient(
                                colors: [scoreColor.opacity(0.8), scoreColor],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * min(score, 1.0))
                }
            }
            .frame(height: 8)
        }
    }
}

// MARK: - AI Response Card

struct AIResponseCard: View {
    let response: String
    let isGenerating: Bool
    let availability: FoundationModelAvailability
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: "apple.intelligence")
                    .foregroundColor(.orange)
                Text("Apple Intelligence Response")
                    .font(.headline)
                Spacer()
                
                if availability == .available {
                    HStack(spacing: 8) {
                        // Copy button
                        if !response.isEmpty {
                            Button(action: copyResponse) {
                                Image(systemName: "doc.on.doc")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .padding(6)
                                    .background(Color(UIColor.tertiarySystemBackground))
                                    .clipShape(Circle())
                            }
                        }
                        
                        if isGenerating {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else if !response.isEmpty {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundColor(.green)
                                .font(.caption)
                        }
                    }
                }
            }
            
            // Show response (streams in progressively)
            if !response.isEmpty {
                Markdown(response)
                    .markdownTheme(.gitHub)
                    .markdownTextStyle(\.code) {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.85))
                        BackgroundColor(.secondary.opacity(0.2))
                    }
                    .markdownBlockStyle(\.codeBlock) { configuration in
                        ScrollView(.horizontal, showsIndicators: false) {
                            configuration.label
                                .padding(8)
                                .background(Color.secondary.opacity(0.2))
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.orange.opacity(0.08))
                    .cornerRadius(12)
                    .animation(.easeInOut(duration: 0.1), value: response)
                    .onLongPressGesture {
                        copyResponse()
                    }
            } else if isGenerating {
                HStack(spacing: 12) {
                    ProgressView()
                    Text("Thinking...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
            } else {
                // Show availability status
                availabilityView
            }
        }
        .padding(20)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .padding(.horizontal)
    }
    
    @ViewBuilder
    var availabilityView: some View {
        switch availability {
        case .available:
            EmptyView()
        case .notEligible:
            HStack {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundColor(.orange)
                Text("This device doesn't support Apple Intelligence")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        case .notEnabled:
            HStack {
                Image(systemName: "gear")
                    .foregroundColor(.orange)
                Text("Please enable Apple Intelligence in Settings")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        case .notReady:
            HStack {
                Image(systemName: "arrow.down.circle")
                    .foregroundColor(.orange)
                Text("Apple Intelligence is downloading...")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        case .unknown:
            HStack {
                Image(systemName: "questionmark.circle")
                    .foregroundColor(.orange)
                Text("Apple Intelligence status unknown")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
            }
        }
    }
    
    private func copyResponse() {
        UIPasteboard.general.string = response
        let generator = UIImpactFeedbackGenerator(style: .medium)
        generator.impactOccurred()
    }
}

#Preview {
    ContentView()
}
