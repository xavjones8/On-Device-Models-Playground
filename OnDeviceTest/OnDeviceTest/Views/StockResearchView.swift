/*
 StockResearchView.swift
 
 Stock research chat interface with tool-calling capabilities.
 Demonstrates Apple Foundation Models' ability to use external tools.
 
 Features:
 - Natural language stock queries
 - Automatic tool invocation (fetch data, compute metrics, etc.)
 - Interactive charts using Swift Charts
 - Tool call visualization (shows what the AI did)
 
 Tools Available:
 - fetchTimeSeries: Get historical price data
 - computeMetrics: Calculate returns, volatility
 - compareStocks: Compare multiple stocks
 - generateChartData: Prepare chart visualization
 
 Architecture:
 - StockResearchViewModel: Manages session with tools
 - StockResearchView: Chat UI with embedded charts
 - Tool call section: Collapsible view of AI's actions
 
 Data Flow:
 1. User asks about stocks
 2. AI calls tools via LanguageModelSession
 3. Tools fetch/compute data, cache results
 4. AI synthesizes response with tool outputs
 5. Charts render from GenerateChartDataTool.lastChartData
 */

import SwiftUI
import FoundationModels
import Charts
import MarkdownUI
import Combine

// MARK: - Stock Research Chat View

struct StockResearchView: View {
    @StateObject private var viewModel = StockResearchViewModel()
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
                                
                                // Welcome message
                                if viewModel.messages.isEmpty {
                                    WelcomeCard()
                                        .padding(.horizontal)
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
                                    StockChatMessageView(message: message)
                                        .id(message.id)
                                }
                            }
                            .padding(.vertical)
                        }
                        .onChange(of: viewModel.messages.count) { _ in
                            scrollToBottom(proxy: proxy)
                        }
                        .onChange(of: viewModel.messages.last?.content ?? "") { _ in
                            scrollToBottom(proxy: proxy)
                        }
                    }
                    
                    // Input area
                    VStack(spacing: 0) {
                        Divider()
                        
                        // Quick prompts
                        if viewModel.messages.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 8) {
                                    QuickPromptButton(text: "Compare NVDA vs AMD") {
                                        viewModel.inputText = "Compare NVIDIA and AMD stock performance over the last year. Show me detailed metrics, which is outperforming, and generate a chart comparing them."
                                    }
                                    QuickPromptButton(text: "Analyze AAPL") {
                                        viewModel.inputText = "Analyze only Apple (AAPL) stock over the last 6 months. Show me returns, volatility, price trends, and generate a chart."
                                    }
                                    QuickPromptButton(text: "Tech comparison") {
                                        viewModel.inputText = "Compare Microsoft, Google, and Meta stock performance over the past year. Give me detailed metrics for each, determine the best performer, and generate a normalized comparison chart."
                                    }
                                }
                                .padding(.horizontal)
                                .padding(.vertical, 8)
                            }
                        }
                        
                        HStack(spacing: 12) {
                            TextField("Ask about stocks...", text: $viewModel.inputText, axis: .vertical)
                                .textFieldStyle(.plain)
                                .padding(.horizontal, 16)
                                .padding(.vertical, 12)
                                .background(Color(UIColor.secondarySystemBackground))
                                .cornerRadius(24)
                                .lineLimit(1...5)
                                .focused($isInputFocused)
                                .disabled(viewModel.isProcessing || viewModel.availability != .available)
                            
                            Button(action: {
                                viewModel.sendMessage()
                                isInputFocused = false
                            }) {
                                Image(systemName: viewModel.isProcessing ? "stop.circle.fill" : "arrow.up.circle.fill")
                                    .font(.system(size: 32))
                                    .foregroundColor(
                                        viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty ||
                                        viewModel.isProcessing ||
                                        viewModel.availability != .available
                                        ? .gray : .green
                                    )
                            }
                            .disabled(
                                viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty ||
                                viewModel.isProcessing ||
                                viewModel.availability != .available
                            )
                        }
                        .padding(.horizontal)
                        .padding(.vertical, 12)
                        .background(Color(UIColor.systemBackground))
                    }
                }
            }
            .navigationTitle("Stock Research")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if !viewModel.messages.isEmpty {
                        Button("Clear") {
                            viewModel.clearSession()
                        }
                        .foregroundColor(.red)
                    }
                }
            }
        }
    }
    
    private func scrollToBottom(proxy: ScrollViewProxy) {
        if let lastMessage = viewModel.messages.last {
            withAnimation(.easeOut(duration: 0.2)) {
                proxy.scrollTo(lastMessage.id, anchor: .bottom)
            }
        }
    }
}

// MARK: - View Model

struct StockChatMessage: Identifiable {
    let id = UUID()
    let isUser: Bool
    var content: String
    var toolCalls: [ToolCallInfo] = []
    var chartData: GenerateChartDataTool.ChartData?
    var isStreaming: Bool = false
}

struct ToolCallInfo: Identifiable {
    let id = UUID()
    let toolName: String
    let description: String
    let isComplete: Bool
}

@MainActor
class StockResearchViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var isProcessing: Bool = false
    @Published var availability: FoundationModelAvailability = .unknown
    @Published var errorMessage: String?
    @Published var messages: [StockChatMessage] = []
    
    private var session: LanguageModelSession?
    private let maxMessagesToKeep = 4 // Keep last 2 exchanges when resetting
    
    private let systemInstructions = """
        You are a stock research analyst.
        
        IMPORTANT: If user asks about ONE stock, analyze that ONE stock. Do NOT ask for more stocks. Do NOT use compareStocks for single stocks.
        
        TOOLS:
        - fetchTimeSeries: Get price data
        - computeMetrics: Calculate returns and volatility (works with 1 stock)
        - compareStocks: ONLY use when user asks about 2+ stocks
        - generateChartData: Create chart (works with 1 stock)
        
        FOR SINGLE STOCK (e.g. "analyze AAPL"):
        1. fetchTimeSeries for that stock
        2. computeMetrics with tickers: [that stock]
        3. generateChartData with tickers: [that stock]
        4. Provide analysis
        
        FOR MULTIPLE STOCKS (e.g. "compare AAPL and MSFT"):
        1. fetchTimeSeries for each stock
        2. computeMetrics for all
        3. compareStocks
        4. generateChartData
        5. Provide analysis
        
        NEVER ask user for additional stocks. Analyze exactly what they asked for.
        """
    
    init() {
        checkAvailability()
        setupSession()
    }
    
    private func checkAvailability() {
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
    
    private func setupSession() {
        guard availability == .available else { return }
        
        session = LanguageModelSession(
            tools: [
                FetchTimeSeriesTool(),
                ComputeMetricsTool(),
                CompareStocksTool(),
                GenerateChartDataTool()
            ],
            instructions: systemInstructions
        )
    }
    
    /// Reset session but preserve recent messages for context continuity
    private func resetSessionWithContext() {
        // Keep only the most recent messages
        let recentMessages = Array(messages.suffix(maxMessagesToKeep))
        
        // Clear tool caches and create new session
        FetchTimeSeriesTool.cachedSeries = [:]
        FetchTimeSeriesTool.toolCallLog = []
        GenerateChartDataTool.lastChartData = nil
        setupSession()
        
        // Restore recent messages (UI only, session is fresh)
        messages = recentMessages
        
        // Add a system note about context reset
        let systemNote = StockChatMessage(
            isUser: false,
            content: "_(Conversation trimmed to stay within limits. Tool data cache cleared.)_"
        )
        messages.insert(systemNote, at: 0)
    }
    
    func clearSession() {
        messages = []
        FetchTimeSeriesTool.cachedSeries = [:]
        FetchTimeSeriesTool.toolCallLog = []
        GenerateChartDataTool.lastChartData = nil
        session = nil  // Explicitly clear old session
        setupSession()
        print("🟢 Stock session cleared and recreated")
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
        
        // Clear previous tool call log
        FetchTimeSeriesTool.toolCallLog = []
        GenerateChartDataTool.lastChartData = nil
        
        // Add user message (only if not a retry - retry reuses existing user message)
        if !isRetry {
            messages.append(StockChatMessage(isUser: true, content: prompt))
        }
        
        isProcessing = true
        errorMessage = nil
        
        // Add AI response placeholder
        let aiMessageIndex = messages.count
        messages.append(StockChatMessage(isUser: false, content: "", isStreaming: true))
        
        Task {
            do {
                let options = GenerationOptions(temperature: 0.7)
                
                // Use respond with tools - use currentSession captured fresh
                let response = try await currentSession.respond(to: prompt, options: options)
                
                // Build tool calls from log
                let toolCalls = FetchTimeSeriesTool.toolCallLog.map { log -> ToolCallInfo in
                    let toolName: String
                    if log.contains("fetchTimeSeries") {
                        toolName = "fetchTimeSeries"
                    } else if log.contains("computeMetrics") {
                        toolName = "computeMetrics"
                    } else if log.contains("compareStocks") {
                        toolName = "compareStocks"
                    } else if log.contains("generateChartData") {
                        toolName = "generateChartData"
                    } else {
                        toolName = "unknown"
                    }
                    return ToolCallInfo(toolName: toolName, description: log, isComplete: true)
                }
                
                // Update the AI message with content, tool calls, and chart
                messages[aiMessageIndex] = StockChatMessage(
                    isUser: false,
                    content: response.content,
                    toolCalls: toolCalls,
                    chartData: GenerateChartDataTool.lastChartData,
                    isStreaming: false
                )
                isProcessing = false
                
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
                
                print("🔴 Stock error: \(errorString)")
                print("🔴 Context overflow detected: \(isContextOverflow), isRetry: \(isRetry)")
                
                // Remove the failed AI placeholder
                if aiMessageIndex < messages.count {
                    messages.remove(at: aiMessageIndex)
                }
                
                if isContextOverflow && !isRetry {
                    // Context overflow - reset and auto-retry
                    // Remove the user message we just added
                    if messages.count > 0 && messages.last?.isUser == true {
                        messages.removeLast()
                    }
                    
                    // Full reset - clear everything and start fresh
                    clearSession()
                    
                    // Show brief status then auto-retry
                    errorMessage = "Context limit reached. Retrying with fresh session..."
                    
                    // Small delay then retry
                    try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 sec
                    errorMessage = nil
                    
                    // Retry with fresh session
                    executePrompt(prompt, isRetry: true)
                    return // Don't set isProcessing = false, executePrompt handles it
                    
                } else if isContextOverflow && isRetry {
                    // Already retried once, give up
                    errorMessage = "Request too large. Try a simpler query."
                    isProcessing = false
                } else if let toolError = error as? LanguageModelSession.ToolCallError {
                    errorMessage = "Tool '\(toolError.tool.name)' failed: \(toolError.underlyingError.localizedDescription)"
                    isProcessing = false
                } else {
                    errorMessage = "Failed: \(error.localizedDescription)"
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - Chat Message View

struct StockChatMessageView: View {
    let message: StockChatMessage
    
    var body: some View {
        HStack {
            if message.isUser {
                Spacer(minLength: 60)
                StockUserBubble(content: message.content)
            } else {
                StockAIBubble(message: message)
                Spacer(minLength: 40)
            }
        }
        .padding(.horizontal)
    }
}

struct StockUserBubble: View {
    let content: String
    
    var body: some View {
        Text(content)
            .font(.body)
            .foregroundColor(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(Color.green)
            .cornerRadius(20)
    }
}

struct StockAIBubble: View {
    let message: StockChatMessage
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Main content
            if message.content.isEmpty && message.isStreaming {
                HStack(spacing: 8) {
                    ProgressView()
                        .scaleEffect(0.8)
                    Text("Analyzing...")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(16)
            } else if !message.content.isEmpty {
                Markdown(message.content)
                    .markdownTheme(.basic)
                    .markdownTextStyle(\.code) {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.85))
                        ForegroundColor(.primary)
                        BackgroundColor(.clear)
                    }
                    .padding(16)
            }
            
            // Embedded chart (if available)
            if let chartData = message.chartData {
                EmbeddedChartView(chartData: chartData)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 8)
            }
            
            // Tool calls at the bottom
            if !message.toolCalls.isEmpty {
                ToolCallsSection(toolCalls: message.toolCalls)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
            }
        }
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
        .onLongPressGesture {
            UIPasteboard.general.string = message.content
            let generator = UIImpactFeedbackGenerator(style: .medium)
            generator.impactOccurred()
        }
    }
}

// MARK: - Tool Calls Section

struct ToolCallsSection: View {
    let toolCalls: [ToolCallInfo]
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button(action: { isExpanded.toggle() }) {
                HStack {
                    Image(systemName: "wrench.and.screwdriver.fill")
                        .font(.caption)
                        .foregroundColor(.orange)
                    Text("\(toolCalls.count) tool\(toolCalls.count == 1 ? "" : "s") called")
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.secondary)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .buttonStyle(.plain)
            
            if isExpanded {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(toolCalls) { call in
                        HStack(spacing: 8) {
                            Image(systemName: iconForTool(call.toolName))
                                .font(.caption2)
                                .foregroundColor(colorForTool(call.toolName))
                                .frame(width: 16)
                            
                            Text(call.description)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                                .lineLimit(2)
                            
                            Spacer()
                            
                            Image(systemName: "checkmark.circle.fill")
                                .font(.caption2)
                                .foregroundColor(.green)
                        }
                        .padding(8)
                        .background(Color(UIColor.tertiarySystemBackground))
                        .cornerRadius(8)
                    }
                }
            }
        }
        .padding(12)
        .background(Color(UIColor.tertiarySystemBackground).opacity(0.5))
        .cornerRadius(12)
    }
    
    func iconForTool(_ name: String) -> String {
        switch name {
        case "fetchTimeSeries": return "arrow.down.doc.fill"
        case "computeMetrics": return "function"
        case "compareStocks": return "arrow.left.arrow.right"
        case "generateChartData": return "chart.xyaxis.line"
        default: return "wrench.fill"
        }
    }
    
    func colorForTool(_ name: String) -> Color {
        switch name {
        case "fetchTimeSeries": return .blue
        case "computeMetrics": return .purple
        case "compareStocks": return .orange
        case "generateChartData": return .green
        default: return .gray
        }
    }
}

// MARK: - Embedded Chart View

struct EmbeddedChartView: View {
    let chartData: GenerateChartDataTool.ChartData
    
    private var isPerformanceChart: Bool {
        ["normalized", "comparison", "performance"].contains(chartData.chartType.lowercased())
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: "chart.xyaxis.line")
                    .foregroundColor(.green)
                    .font(.caption)
                Text(isPerformanceChart ? "Performance Chart" : "Price Chart")
                    .font(.caption)
                    .fontWeight(.medium)
                Spacer()
                Text(chartData.tickers.joined(separator: " vs "))
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            
            // Chart
            if isPerformanceChart {
                NormalizedChartEmbed(data: chartData.normalizedSeries)
            } else {
                PriceChartEmbed(data: chartData.series)
            }
        }
        .padding(12)
        .background(Color(UIColor.tertiarySystemBackground))
        .cornerRadius(12)
    }
}

// Helper struct for chart data points
struct ChartDataPoint: Identifiable {
    let id = UUID()
    let ticker: String
    let date: Date
    let value: Double
}

struct NormalizedChartEmbed: View {
    let data: [String: [(date: Date, value: Double)]]
    
    private let colors: [Color] = [.blue, .orange, .green, .purple, .red]
    
    // Convert dictionary to flat array of identifiable points
    private var chartPoints: [ChartDataPoint] {
        var points: [ChartDataPoint] = []
        for (ticker, series) in data {
            let sorted = series.sorted { $0.date < $1.date }
            for point in sorted {
                points.append(ChartDataPoint(ticker: ticker, date: point.date, value: point.value))
            }
        }
        return points
    }
    
    private var sortedTickers: [String] {
        data.keys.sorted()
    }
    
    var body: some View {
        Chart(chartPoints) { point in
            LineMark(
                x: .value("Date", point.date),
                y: .value("Change %", point.value),
                series: .value("Ticker", point.ticker)
            )
            .foregroundStyle(by: .value("Ticker", point.ticker))
        }
        .chartForegroundStyleScale(domain: sortedTickers, range: Array(colors.prefix(sortedTickers.count)))
        .chartYAxis {
            AxisMarks(position: .leading) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let val = value.as(Double.self) {
                        Text("\(val >= 0 ? "+" : "")\(Int(val))%")
                            .font(.caption2)
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                AxisGridLine()
                AxisValueLabel(format: .dateTime.month(.abbreviated))
            }
        }
        .chartLegend(position: .bottom, spacing: 8)
        .frame(height: 180)
        .overlay {
            // Zero baseline
            GeometryReader { geo in
                let yScale = geo.size.height
                let allValues = chartPoints.map { $0.value }
                let minVal = allValues.min() ?? 0
                let maxVal = allValues.max() ?? 0
                let range = maxVal - minVal
                if range > 0 && minVal < 0 && maxVal > 0 {
                    let zeroY = geo.size.height * (maxVal / range)
                    Path { path in
                        path.move(to: CGPoint(x: 0, y: zeroY))
                        path.addLine(to: CGPoint(x: geo.size.width, y: zeroY))
                    }
                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [5, 5]))
                    .foregroundColor(.gray.opacity(0.5))
                }
            }
        }
    }
}

struct PriceChartEmbed: View {
    let data: [String: [(date: Date, price: Double)]]
    
    private let colors: [Color] = [.blue, .orange, .green, .purple, .red]
    
    // Convert dictionary to flat array of identifiable points
    private var chartPoints: [ChartDataPoint] {
        var points: [ChartDataPoint] = []
        for (ticker, series) in data {
            let sorted = series.sorted { $0.date < $1.date }
            for point in sorted {
                points.append(ChartDataPoint(ticker: ticker, date: point.date, value: point.price))
            }
        }
        return points
    }
    
    private var sortedTickers: [String] {
        data.keys.sorted()
    }
    
    var body: some View {
        Chart(chartPoints) { point in
            LineMark(
                x: .value("Date", point.date),
                y: .value("Price", point.value),
                series: .value("Ticker", point.ticker)
            )
            .foregroundStyle(by: .value("Ticker", point.ticker))
        }
        .chartForegroundStyleScale(domain: sortedTickers, range: Array(colors.prefix(sortedTickers.count)))
        .chartYAxis {
            AxisMarks(position: .leading) { value in
                AxisGridLine()
                AxisValueLabel {
                    if let val = value.as(Double.self) {
                        Text("$\(Int(val))")
                            .font(.caption2)
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 4)) { _ in
                AxisGridLine()
                AxisValueLabel(format: .dateTime.month(.abbreviated))
            }
        }
        .chartLegend(position: .bottom, spacing: 8)
        .frame(height: 180)
    }
}

// MARK: - Supporting Views

struct WelcomeCard: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(.system(size: 48))
                .foregroundColor(.green)
            
            Text("Stock Research Analyst")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Ask me to analyze stocks, compare performance, and identify trends. I can fetch real market data and provide insights.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            
            VStack(alignment: .leading, spacing: 8) {
                FeatureRow(icon: "chart.bar.fill", text: "Fetch historical price data")
                FeatureRow(icon: "percent", text: "Calculate returns & volatility")
                FeatureRow(icon: "arrow.left.arrow.right", text: "Compare multiple stocks")
                FeatureRow(icon: "chart.xyaxis.line", text: "Visualize trends")
            }
            .padding(.top, 8)
        }
        .padding(24)
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(16)
    }
}

struct FeatureRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .foregroundColor(.green)
                .frame(width: 24)
            Text(text)
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }
}

struct QuickPromptButton: View {
    let text: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            Text(text)
                .font(.caption)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.green.opacity(0.15))
                .foregroundColor(.green)
                .cornerRadius(16)
        }
    }
}

#Preview {
    StockResearchView()
}
