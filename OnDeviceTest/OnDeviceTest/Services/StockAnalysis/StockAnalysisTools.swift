/*
 StockAnalysisTools.swift
 
 Tool definitions for Apple Foundation Models tool-calling API.
 These tools enable the on-device LLM to perform stock analysis.
 
 Tools Defined:
 1. FetchTimeSeriesTool: Fetches historical price data from AlphaVantage
 2. ComputeMetricsTool: Calculates financial metrics (returns, volatility)
 3. CompareStocksTool: Compares performance between multiple stocks
 4. GenerateChartDataTool: Prepares data for Swift Charts visualization
 
 Tool Calling Flow:
 1. User asks about stocks (e.g., "Compare NVDA vs AMD")
 2. LLM determines which tools to call and with what arguments
 3. Each tool executes and returns a string result
 4. LLM synthesizes tool outputs into a coherent response
 
 Data Flow:
 - FetchTimeSeriesTool caches data in static var cachedSeries
 - Other tools read from this cache (no re-fetching)
 - GenerateChartDataTool stores chart data for UI rendering
 
 Logging:
 - All tool calls are logged to toolCallLog for UI display
 - Helps users understand what the AI is doing
 
 Usage with LanguageModelSession:
   let session = LanguageModelSession(
       tools: [FetchTimeSeriesTool(), ComputeMetricsTool(), ...],
       instructions: "You are a stock analyst..."
   )
   let response = try await session.respond(to: userPrompt)
 */

import Foundation
import FoundationModels

// MARK: - Fetch Time Series Tool

struct FetchTimeSeriesTool: Tool {
    let name = "fetchTimeSeries"
    let description = "Fetches historical stock price data for a given ticker symbol. Returns daily data for 1-3 month ranges, weekly data for 6+ month ranges to ensure full coverage."
    
    @Generable
    struct Arguments {
        @Guide(description: "The stock ticker symbol (e.g., NVDA, AAPL, AMD)")
        var ticker: String
        
        @Guide(description: "Time range: 1m (1 month), 3m, 6m, 1y (1 year), or 5y")
        var range: String
    }
    
    // Store fetched data for use by other tools
    static var cachedSeries: [String: StockTimeSeries] = [:]
    
    // Track tool calls for UI display
    static var toolCallLog: [String] = []
    
    func call(arguments: Arguments) async throws -> String {
        let logEntry = "🔧 fetchTimeSeries(ticker: \"\(arguments.ticker)\", range: \"\(arguments.range)\")"
        print(logEntry)
        FetchTimeSeriesTool.toolCallLog.append(logEntry)
        
        let service = AlphaVantageService.shared
        
        do {
            let series = try await service.fetchTimeSeries(ticker: arguments.ticker, range: arguments.range)
            
            // Cache for later use
            FetchTimeSeriesTool.cachedSeries[arguments.ticker.uppercased()] = series
            
            guard !series.dataPoints.isEmpty else {
                let result = "No data available for \(arguments.ticker) in the specified range."
                print("   ❌ \(result)")
                return result
            }
            
            let firstPrice = series.dataPoints.first?.close ?? 0
            let lastPrice = series.dataPoints.last?.close ?? 0
            let priceChange = lastPrice - firstPrice
            let percentChange = firstPrice > 0 ? (priceChange / firstPrice) * 100 : 0
            
            let dateFormatter = DateFormatter()
            dateFormatter.dateStyle = .medium
            
            let startDate = series.dataPoints.first.map { dateFormatter.string(from: $0.date) } ?? "N/A"
            let endDate = series.dataPoints.last.map { dateFormatter.string(from: $0.date) } ?? "N/A"
            
            // Determine if this is likely weekly or daily data based on point count and range
            let dataFrequency = arguments.range.lowercased().contains("y") || 
                               arguments.range.lowercased().contains("6m") ? "weekly" : "daily"
            
            let result = """
            Successfully fetched \(series.dataPoints.count) \(dataFrequency) data points for \(arguments.ticker.uppercased()):
            - Period: \(startDate) to \(endDate)
            - Starting Price: $\(String(format: "%.2f", firstPrice))
            - Current Price: $\(String(format: "%.2f", lastPrice))
            - Change: \(priceChange >= 0 ? "+" : "")\(String(format: "%.2f", priceChange)) (\(percentChange >= 0 ? "+" : "")\(String(format: "%.1f", percentChange))%)
            - Period High: $\(String(format: "%.2f", series.prices.max() ?? 0))
            - Period Low: $\(String(format: "%.2f", series.prices.min() ?? 0))
            """
            print("   ✅ Fetched \(series.dataPoints.count) data points")
            return result
        } catch {
            let result = "Failed to fetch data for \(arguments.ticker): \(error.localizedDescription)"
            print("   ❌ \(result)")
            return result
        }
    }
}

// MARK: - Compute Metrics Tool

struct ComputeMetricsTool: Tool {
    let name = "computeMetrics"
    let description = "Computes financial metrics for previously fetched stock data including mean return, volatility, and optionally correlation between two stocks."
    
    @Generable
    struct Arguments {
        @Guide(description: "List of metrics to compute: mean_return, volatility, total_return, correlation")
        var metrics: [String]
        
        @Guide(description: "List of ticker symbols to analyze (must have been fetched first)")
        var tickers: [String]
    }
    
    func call(arguments: Arguments) async throws -> String {
        let logEntry = "🔧 computeMetrics(metrics: \(arguments.metrics), tickers: \(arguments.tickers))"
        print(logEntry)
        FetchTimeSeriesTool.toolCallLog.append(logEntry)
        
        let service = AlphaVantageService.shared
        var results: [String] = []
        
        // Get cached series
        var seriesData: [String: StockTimeSeries] = [:]
        for ticker in arguments.tickers {
            let upperTicker = ticker.uppercased().replacingOccurrences(of: "$", with: "")
            if let cached = FetchTimeSeriesTool.cachedSeries[upperTicker] {
                seriesData[upperTicker] = cached
            } else {
                results.append("⚠️ No data found for \(upperTicker). Please fetch it first using fetchTimeSeries.")
            }
        }
        
        guard !seriesData.isEmpty else {
            print("   ❌ No stock data available")
            return "No stock data available. Please fetch time series data first."
        }
        
        // Compute metrics for each ticker
        for (ticker, series) in seriesData.sorted(by: { $0.key < $1.key }) {
            let metrics = service.computeMetrics(series: series)
            var tickerResults: [String] = ["📊 **\(ticker)** Metrics:"]
            
            // Always show the data period for clarity
            tickerResults.append("  - Data Period: \(metrics.formattedPeriod) (\(metrics.periodDays) days)")
            
            for metric in arguments.metrics {
                switch metric.lowercased() {
                case "mean_return", "annualized_return", "return":
                    let periodPct = metrics.periodReturn * 100
                    let annualPct = metrics.annualizedReturn * 100
                    tickerResults.append("  - Period Return: \(periodPct >= 0 ? "+" : "")\(String(format: "%.2f", periodPct))%")
                    tickerResults.append("  - Annualized Return (CAGR): \(annualPct >= 0 ? "+" : "")\(String(format: "%.2f", annualPct))%")
                case "volatility":
                    let volPct = metrics.volatility * 100
                    tickerResults.append("  - Annualized Volatility: \(String(format: "%.2f", volPct))%")
                case "total_return", "period_return":
                    let totalPct = metrics.periodReturn * 100
                    tickerResults.append("  - Period Return: \(totalPct >= 0 ? "+" : "")\(String(format: "%.2f", totalPct))%")
                default:
                    break
                }
            }
            
            tickerResults.append("  - Current Price: $\(String(format: "%.2f", metrics.currentPrice))")
            tickerResults.append("  - Period High/Low: $\(String(format: "%.2f", metrics.minPrice)) - $\(String(format: "%.2f", metrics.maxPrice))")
            
            results.append(tickerResults.joined(separator: "\n"))
        }
        
        // Compute correlation if requested and we have 2+ tickers
        if arguments.metrics.contains(where: { $0.lowercased() == "correlation" }) && seriesData.count >= 2 {
            let tickers = Array(seriesData.keys).sorted()
            for i in 0..<tickers.count {
                for j in (i+1)..<tickers.count {
                    let ticker1 = tickers[i]
                    let ticker2 = tickers[j]
                    if let series1 = seriesData[ticker1], let series2 = seriesData[ticker2] {
                        let correlation = service.computeCorrelation(series1: series1, series2: series2)
                        let interpretation: String
                        if correlation > 0.7 {
                            interpretation = "strongly correlated (move together)"
                        } else if correlation > 0.3 {
                            interpretation = "moderately correlated"
                        } else if correlation > -0.3 {
                            interpretation = "weakly correlated"
                        } else if correlation > -0.7 {
                            interpretation = "moderately inversely correlated"
                        } else {
                            interpretation = "strongly inversely correlated (move opposite)"
                        }
                        results.append("\n📈 **Correlation** between \(ticker1) and \(ticker2): \(String(format: "%.3f", correlation)) (\(interpretation))")
                    }
                }
            }
        }
        
        return results.joined(separator: "\n\n")
    }
}

// MARK: - Compare Stocks Tool

struct CompareStocksTool: Tool {
    let name = "compareStocks"
    let description = "Compares the performance of two or more stocks, determining which is outperforming or if they track similarly."
    
    @Generable
    struct Arguments {
        @Guide(description: "List of ticker symbols to compare")
        var tickers: [String]
    }
    
    func call(arguments: Arguments) async throws -> String {
        let logEntry = "🔧 compareStocks(tickers: \(arguments.tickers))"
        print(logEntry)
        FetchTimeSeriesTool.toolCallLog.append(logEntry)
        
        let service = AlphaVantageService.shared
        var metricsData: [(ticker: String, metrics: StockMetrics)] = []
        
        for ticker in arguments.tickers {
            let upperTicker = ticker.uppercased().replacingOccurrences(of: "$", with: "")
            if let series = FetchTimeSeriesTool.cachedSeries[upperTicker] {
                let metrics = service.computeMetrics(series: series)
                metricsData.append((ticker: upperTicker, metrics: metrics))
            }
        }
        
        guard metricsData.count >= 2 else {
            print("   ❌ Need at least 2 stocks with fetched data")
            return "Need at least 2 stocks with fetched data to compare. Please fetch time series data first."
        }
        
        // Sort by period return
        let sorted = metricsData.sorted { $0.metrics.periodReturn > $1.metrics.periodReturn }
        let best = sorted.first!
        let worst = sorted.last!
        
        var result = "## Stock Comparison Analysis\n\n"
        
        // Data period info
        if let firstMetrics = metricsData.first?.metrics {
            result += "**Analysis Period:** \(firstMetrics.formattedPeriod) (\(firstMetrics.periodDays) trading days)\n\n"
        }
        
        // Performance ranking
        result += "### Performance Ranking (by Period Return):\n"
        for (index, data) in sorted.enumerated() {
            let returnPct = data.metrics.periodReturn * 100
            let annualPct = data.metrics.annualizedReturn * 100
            let emoji = index == 0 ? "🥇" : (index == 1 ? "🥈" : "🥉")
            result += "\(emoji) **\(data.ticker)**: \(returnPct >= 0 ? "+" : "")\(String(format: "%.1f", returnPct))% (CAGR: \(annualPct >= 0 ? "+" : "")\(String(format: "%.1f", annualPct))%)\n"
        }
        
        // Price info
        result += "\n### Current Prices:\n"
        for data in sorted {
            result += "- **\(data.ticker)**: $\(String(format: "%.2f", data.metrics.currentPrice))\n"
        }
        
        // Comparison
        result += "\n### Analysis:\n"
        
        let returnDiff = (best.metrics.periodReturn - worst.metrics.periodReturn) * 100
        if returnDiff > 20 {
            result += "**\(best.ticker) is significantly outperforming \(worst.ticker)** with a \(String(format: "%.1f", returnDiff)) percentage point advantage over this period.\n"
        } else if returnDiff > 5 {
            result += "**\(best.ticker) is moderately outperforming \(worst.ticker)** with a \(String(format: "%.1f", returnDiff)) percentage point advantage.\n"
        } else {
            result += "**\(best.ticker) and \(worst.ticker) are tracking similarly**, with only a \(String(format: "%.1f", returnDiff)) percentage point difference.\n"
        }
        
        // Volatility comparison
        result += "\n### Risk (Annualized Volatility):\n"
        for data in sorted {
            let vol = data.metrics.volatility * 100
            let riskLevel = vol > 50 ? "High" : (vol > 30 ? "Moderate" : "Low")
            result += "- **\(data.ticker)**: \(String(format: "%.1f", vol))% (\(riskLevel) risk)\n"
        }
        
        // Correlation if we have exactly 2 stocks
        if metricsData.count == 2 {
            let series1 = FetchTimeSeriesTool.cachedSeries[metricsData[0].ticker]!
            let series2 = FetchTimeSeriesTool.cachedSeries[metricsData[1].ticker]!
            let correlation = service.computeCorrelation(series1: series1, series2: series2)
            
            result += "\n### Correlation: \(String(format: "%.3f", correlation))\n"
            if correlation > 0.7 {
                result += "These stocks are **highly correlated** and tend to move together. They may not provide much diversification benefit when held together."
            } else if correlation > 0.3 {
                result += "These stocks have **moderate correlation**. They share some common price movements but also have independent factors."
            } else {
                result += "These stocks have **low correlation**. They can provide good diversification when held together."
            }
        }
        
        return result
    }
}

// MARK: - Generate Chart Data Tool

struct GenerateChartDataTool: Tool {
    let name = "generateChartData"
    let description = "Creates chart for 1 or more stocks. Works with a single stock or multiple stocks. Use 'price' type for dollar prices, 'performance' type for % change."
    
    @Generable
    struct Arguments {
        @Guide(description: "Ticker symbols to chart (can be just one stock, e.g. ['AAPL'], or multiple)")
        var tickers: [String]
        
        @Guide(description: "Chart type: 'price' for dollar values ($), 'performance' for % change from period start")
        var chartType: String
    }
    
    // This will be used by the UI to render charts
    static var lastChartData: ChartData?
    
    struct ChartData {
        let tickers: [String]
        let chartType: String
        let series: [String: [(date: Date, price: Double)]]
        let normalizedSeries: [String: [(date: Date, value: Double)]]
    }
    
    func call(arguments: Arguments) async throws -> String {
        let logEntry = "🔧 generateChartData(tickers: \(arguments.tickers), chartType: \"\(arguments.chartType)\")"
        print(logEntry)
        FetchTimeSeriesTool.toolCallLog.append(logEntry)
        
        var series: [String: [(date: Date, price: Double)]] = [:]
        var normalizedSeries: [String: [(date: Date, value: Double)]] = [:]
        
        for ticker in arguments.tickers {
            let upperTicker = ticker.uppercased().replacingOccurrences(of: "$", with: "")
            if let cached = FetchTimeSeriesTool.cachedSeries[upperTicker] {
                let points = cached.dataPoints.map { (date: $0.date, price: $0.close) }
                series[upperTicker] = points
                
                // Normalize to percentage change from start
                if let firstPrice = points.first?.price, firstPrice > 0 {
                    let normalized = points.map { (date: $0.date, value: (($0.price - firstPrice) / firstPrice) * 100) }
                    normalizedSeries[upperTicker] = normalized
                }
            }
        }
        
        guard !series.isEmpty else {
            print("   ❌ No chart data available")
            return "No chart data available. Please fetch time series data first."
        }
        
        // Store for UI rendering
        GenerateChartDataTool.lastChartData = ChartData(
            tickers: Array(series.keys).sorted(),
            chartType: arguments.chartType,
            series: series,
            normalizedSeries: normalizedSeries
        )
        
        var result = "📊 **Chart Data Prepared**\n\n"
        result += "**Chart type:** \(arguments.chartType)\n"
        result += "**Stocks:** \(series.keys.sorted().joined(separator: ", "))\n\n"
        
        // Include summary data points for the response
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        
        let isPerformanceChart = ["normalized", "comparison", "performance"].contains(arguments.chartType.lowercased())
        
        if isPerformanceChart {
            result += "### Performance Summary (% change from start):\n\n"
            for ticker in series.keys.sorted() {
                if let data = normalizedSeries[ticker], let lastPoint = data.last, let firstPoint = data.first {
                    let startDate = dateFormatter.string(from: firstPoint.date)
                    let endDate = dateFormatter.string(from: lastPoint.date)
                    result += "**\(ticker):** \(lastPoint.value >= 0 ? "+" : "")\(String(format: "%.2f", lastPoint.value))%\n"
                    result += "- Period: \(startDate) to \(endDate)\n"
                    result += "- Data points: \(data.count)\n\n"
                }
            }
        } else {
            result += "### Price Summary:\n\n"
            for ticker in series.keys.sorted() {
                if let data = series[ticker], let lastPoint = data.last, let firstPoint = data.first {
                    let startDate = dateFormatter.string(from: firstPoint.date)
                    let endDate = dateFormatter.string(from: lastPoint.date)
                    let minPrice = data.map { $0.price }.min() ?? 0
                    let maxPrice = data.map { $0.price }.max() ?? 0
                    result += "**\(ticker):**\n"
                    result += "- Current: $\(String(format: "%.2f", lastPoint.price))\n"
                    result += "- Period: \(startDate) to \(endDate)\n"
                    result += "- Range: $\(String(format: "%.2f", minPrice)) - $\(String(format: "%.2f", maxPrice))\n"
                    result += "- Data points: \(data.count)\n\n"
                }
            }
        }
        
        // Include sample data points for chart rendering (first, middle, last)
        result += "### Sample Data Points (for chart):\n\n"
        result += "```\n"
        for ticker in series.keys.sorted() {
            if let data = normalizedSeries[ticker], data.count >= 3 {
                let first = data.first!
                let mid = data[data.count / 2]
                let last = data.last!
                result += "\(ticker):\n"
                result += "  \(dateFormatter.string(from: first.date)): \(String(format: "%.2f", first.value))%\n"
                result += "  \(dateFormatter.string(from: mid.date)): \(String(format: "%.2f", mid.value))%\n"
                result += "  \(dateFormatter.string(from: last.date)): \(String(format: "%.2f", last.value))%\n"
            }
        }
        result += "```\n"
        
        result += "\n✅ *Chart data is ready for visualization in the UI.*"
        
        return result
    }
}

