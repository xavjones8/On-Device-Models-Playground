/*
 APIDebugView.swift
 
 Debug interface for testing AlphaVantage API calls directly.
 Useful for verifying API connectivity and data accuracy.
 
 Features:
 - Direct API testing without AI involvement
 - Raw JSON response display
 - Parsed data visualization
 - Metric computation verification
 - Chart preview
 
 Use Cases:
 - Verify API key is working
 - Check rate limit status
 - Debug data parsing issues
 - Validate metric calculations
 - Test different time ranges
 
 Note: This tab is for development/debugging.
 Consider hiding in production builds.
 */

import SwiftUI
import Charts
import Combine

struct APIDebugView: View {
    @StateObject private var viewModel = APIDebugViewModel()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // API Status Card
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "network")
                                .foregroundColor(.blue)
                            Text("AlphaVantage API Debug")
                                .font(.headline)
                        }
                        
                        HStack {
                            Text("API Key:")
                                .foregroundColor(.secondary)
                            Text(viewModel.maskedApiKey)
                                .font(.system(.caption, design: .monospaced))
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Color(UIColor.secondarySystemBackground))
                    .cornerRadius(12)
                    .padding(.horizontal)
                    
                    // Test Fetch Time Series
                    TestCard(
                        title: "fetchTimeSeries",
                        icon: "arrow.down.doc.fill",
                        color: .green
                    ) {
                        VStack(alignment: .leading, spacing: 12) {
                            HStack {
                                TextField("Ticker", text: $viewModel.testTicker)
                                    .textFieldStyle(.roundedBorder)
                                    .autocapitalization(.allCharacters)
                                
                                Picker("Range", selection: $viewModel.testRange) {
                                    Text("1M").tag("1m")
                                    Text("3M").tag("3m")
                                    Text("6M").tag("6m")
                                    Text("1Y").tag("1y")
                                }
                                .pickerStyle(.segmented)
                            }
                            
                            Button(action: { viewModel.testFetchTimeSeries() }) {
                                HStack {
                                    if viewModel.isFetching {
                                        ProgressView()
                                            .scaleEffect(0.8)
                                    }
                                    Text(viewModel.isFetching ? "Fetching..." : "Test Fetch")
                                }
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 10)
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(8)
                            }
                            .disabled(viewModel.isFetching || viewModel.testTicker.isEmpty)
                            
                            if let result = viewModel.fetchResult {
                                ResultView(result: result)
                            }
                        }
                    }
                    
                    // Test Compute Metrics
                    TestCard(
                        title: "computeMetrics",
                        icon: "function",
                        color: .orange
                    ) {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Requires fetched data first")
                                .font(.caption)
                                .foregroundColor(.secondary)
                            
                            if !viewModel.cachedTickers.isEmpty {
                                Text("Available: \(viewModel.cachedTickers.joined(separator: ", "))")
                                    .font(.caption)
                                    .foregroundColor(.green)
                            }
                            
                            Button(action: { viewModel.testComputeMetrics() }) {
                                Text("Compute Metrics")
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 10)
                                    .background(viewModel.cachedTickers.isEmpty ? Color.gray : Color.orange)
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                            }
                            .disabled(viewModel.cachedTickers.isEmpty)
                            
                            if let result = viewModel.metricsResult {
                                ResultView(result: result)
                            }
                        }
                    }
                    
                    // Raw API Response
                    if let rawResponse = viewModel.rawApiResponse {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "doc.text")
                                    .foregroundColor(.purple)
                                Text("Raw API Response")
                                    .font(.headline)
                                Spacer()
                                Button("Copy") {
                                    UIPasteboard.general.string = rawResponse
                                }
                                .font(.caption)
                            }
                            
                            ScrollView(.horizontal, showsIndicators: true) {
                                Text(rawResponse)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundColor(.secondary)
                            }
                            .frame(maxHeight: 200)
                        }
                        .padding()
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    }
                    
                    // Chart Preview
                    if !viewModel.chartData.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "chart.xyaxis.line")
                                    .foregroundColor(.blue)
                                Text("Price Chart")
                                    .font(.headline)
                            }
                            
                            Chart {
                                ForEach(viewModel.chartData, id: \.date) { point in
                                    LineMark(
                                        x: .value("Date", point.date),
                                        y: .value("Price", point.close)
                                    )
                                    .foregroundStyle(.blue)
                                }
                            }
                            .frame(height: 200)
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
                        }
                        .padding()
                        .background(Color(UIColor.secondarySystemBackground))
                        .cornerRadius(12)
                        .padding(.horizontal)
                    }
                    
                    Spacer(minLength: 40)
                }
                .padding(.top)
            }
            .navigationTitle("API Debug")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Clear") {
                        viewModel.clearAll()
                    }
                }
            }
        }
    }
}

struct TestCard<Content: View>: View {
    let title: String
    let icon: String
    let color: Color
    @ViewBuilder let content: () -> Content
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.headline)
            }
            
            content()
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
        .padding(.horizontal)
    }
}

struct ResultView: View {
    let result: APIDebugViewModel.TestResult
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(result.success ? .green : .red)
                Text(result.success ? "Success" : "Failed")
                    .font(.subheadline)
                    .fontWeight(.medium)
            }
            
            Text(result.message)
                .font(.caption)
                .foregroundColor(.secondary)
            
            if let details = result.details {
                Text(details)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundColor(.secondary)
                    .padding(8)
                    .background(Color(UIColor.tertiarySystemBackground))
                    .cornerRadius(6)
            }
        }
        .padding(12)
        .background(result.success ? Color.green.opacity(0.1) : Color.red.opacity(0.1))
        .cornerRadius(8)
    }
}

@MainActor
class APIDebugViewModel: ObservableObject {
    @Published var testTicker: String = "IBM"
    @Published var testRange: String = "1m"
    @Published var isFetching: Bool = false
    @Published var fetchResult: TestResult?
    @Published var metricsResult: TestResult?
    @Published var rawApiResponse: String?
    @Published var chartData: [StockDataPoint] = []
    @Published var cachedTickers: [String] = []
    
    struct TestResult {
        let success: Bool
        let message: String
        let details: String?
    }
    
    var maskedApiKey: String {
        let key = ProcessInfo.processInfo.environment["ALPHAVANTAGE_API_KEY"] ?? "LZBDLIFLMGCPJ5UC"
        if key.count > 8 {
            return String(key.prefix(4)) + "****" + String(key.suffix(4))
        }
        return "****"
    }
    
    func testFetchTimeSeries() {
        isFetching = true
        fetchResult = nil
        rawApiResponse = nil
        chartData = []
        
        Task {
            do {
                // First, let's make a raw API call to see the response
                let apiKey = ProcessInfo.processInfo.environment["ALPHAVANTAGE_API_KEY"] ?? "LZBDLIFLMGCPJ5UC"
                let urlString = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=\(testTicker.uppercased())&apikey=\(apiKey)"
                
                print("🌐 API URL: \(urlString)")
                
                guard let url = URL(string: urlString) else {
                    fetchResult = TestResult(success: false, message: "Invalid URL", details: urlString)
                    isFetching = false
                    return
                }
                
                let (data, response) = try await URLSession.shared.data(from: url)
                
                // Store raw response
                if let jsonString = String(data: data, encoding: .utf8) {
                    rawApiResponse = String(jsonString.prefix(2000))
                    print("📥 Raw Response: \(jsonString.prefix(500))...")
                }
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    fetchResult = TestResult(success: false, message: "Invalid HTTP response", details: nil)
                    isFetching = false
                    return
                }
                
                print("📊 HTTP Status: \(httpResponse.statusCode)")
                
                // Check for API errors in response
                if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    if let errorMessage = json["Error Message"] as? String {
                        fetchResult = TestResult(success: false, message: "API Error", details: errorMessage)
                        isFetching = false
                        return
                    }
                    if let note = json["Note"] as? String {
                        fetchResult = TestResult(success: false, message: "API Rate Limit", details: note)
                        isFetching = false
                        return
                    }
                    if let info = json["Information"] as? String {
                        fetchResult = TestResult(success: false, message: "API Info", details: info)
                        isFetching = false
                        return
                    }
                }
                
                // Now try the actual service
                let service = AlphaVantageService.shared
                let series = try await service.fetchTimeSeries(ticker: testTicker, range: testRange)
                
                // Update cached tickers
                FetchTimeSeriesTool.cachedSeries[testTicker.uppercased()] = series
                cachedTickers = Array(FetchTimeSeriesTool.cachedSeries.keys).sorted()
                
                // Update chart data
                chartData = series.dataPoints
                
                let details = """
                Data Points: \(series.dataPoints.count)
                First Date: \(series.dataPoints.first?.date.formatted() ?? "N/A")
                Last Date: \(series.dataPoints.last?.date.formatted() ?? "N/A")
                Price Range: $\(String(format: "%.2f", series.prices.min() ?? 0)) - $\(String(format: "%.2f", series.prices.max() ?? 0))
                """
                
                fetchResult = TestResult(
                    success: true,
                    message: "Fetched \(series.dataPoints.count) data points for \(testTicker.uppercased())",
                    details: details
                )
                
            } catch {
                print("❌ Error: \(error)")
                fetchResult = TestResult(
                    success: false,
                    message: "Error: \(error.localizedDescription)",
                    details: String(describing: error)
                )
            }
            
            isFetching = false
        }
    }
    
    func testComputeMetrics() {
        guard let ticker = cachedTickers.first,
              let series = FetchTimeSeriesTool.cachedSeries[ticker] else {
            metricsResult = TestResult(success: false, message: "No cached data", details: nil)
            return
        }
        
        let service = AlphaVantageService.shared
        let metrics = service.computeMetrics(series: series)
        
        let details = """
        Ticker: \(metrics.ticker)
        Data Period: \(metrics.formattedPeriod)
        Period Days: \(metrics.periodDays)
        
        Period Return: \(String(format: "%.2f%%", metrics.periodReturn * 100))
        Annualized Return (CAGR): \(String(format: "%.2f%%", metrics.annualizedReturn * 100))
        Annualized Volatility: \(String(format: "%.2f%%", metrics.volatility * 100))
        
        Current Price: $\(String(format: "%.2f", metrics.currentPrice))
        Period Low: $\(String(format: "%.2f", metrics.minPrice))
        Period High: $\(String(format: "%.2f", metrics.maxPrice))
        """
        
        metricsResult = TestResult(
            success: true,
            message: "Computed metrics for \(ticker)",
            details: details
        )
    }
    
    func clearAll() {
        fetchResult = nil
        metricsResult = nil
        rawApiResponse = nil
        chartData = []
        FetchTimeSeriesTool.cachedSeries = [:]
        cachedTickers = []
    }
}

#Preview {
    APIDebugView()
}

