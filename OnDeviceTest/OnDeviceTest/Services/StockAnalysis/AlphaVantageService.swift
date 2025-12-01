/*
 AlphaVantageService.swift
 
 Stock market data service using the AlphaVantage API.
 https://www.alphavantage.co/documentation/
 
 Features:
 - Fetch historical price data (daily/weekly)
 - Compute financial metrics (returns, volatility, CAGR)
 - Calculate correlation between stocks
 
 API Endpoints Used:
 - TIME_SERIES_DAILY: Daily OHLCV data (100 points on free tier)
 - TIME_SERIES_WEEKLY: Weekly data (full history)
 
 Rate Limits (Free Tier):
 - 5 API calls per minute
 - 500 API calls per day
 
 Financial Metrics:
 - Period Return: (end - start) / start
 - CAGR: (1 + return)^(365/days) - 1
 - Volatility: Annualized standard deviation of returns
 - Correlation: Pearson correlation coefficient
 
 Usage:
   let service = AlphaVantageService.shared
   let series = try await service.fetchTimeSeries(ticker: "AAPL", range: "1y")
   let metrics = service.computeMetrics(series: series)
 */

import Foundation

// MARK: - AlphaVantage API Response Models

struct AlphaVantageTimeSeries: Decodable {
    let metaData: MetaData?
    let timeSeries: [String: DailyData]
    
    enum CodingKeys: String, CodingKey {
        case metaData = "Meta Data"
        case timeSeriesDaily = "Time Series (Daily)"
        case timeSeriesWeekly = "Weekly Time Series"
        case timeSeriesMonthly = "Monthly Time Series"
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        metaData = try? container.decode(MetaData.self, forKey: .metaData)
        
        // Try different time series keys
        if let daily = try? container.decode([String: DailyData].self, forKey: .timeSeriesDaily) {
            timeSeries = daily
        } else if let weekly = try? container.decode([String: DailyData].self, forKey: .timeSeriesWeekly) {
            timeSeries = weekly
        } else if let monthly = try? container.decode([String: DailyData].self, forKey: .timeSeriesMonthly) {
            timeSeries = monthly
        } else {
            timeSeries = [:]
        }
    }
}

struct MetaData: Codable {
    let symbol: String?
    let lastRefreshed: String?
    
    enum CodingKeys: String, CodingKey {
        case symbol = "2. Symbol"
        case lastRefreshed = "3. Last Refreshed"
    }
}

struct DailyData: Codable {
    let open: String
    let high: String
    let low: String
    let close: String
    let volume: String
    
    enum CodingKeys: String, CodingKey {
        case open = "1. open"
        case high = "2. high"
        case low = "3. low"
        case close = "4. close"
        case volume = "5. volume"
    }
}

// MARK: - Processed Data Models

struct StockDataPoint: Identifiable {
    let id = UUID()
    let date: Date
    let close: Double
    let volume: Int
}

struct StockTimeSeries {
    let ticker: String
    let dataPoints: [StockDataPoint]
    
    var prices: [Double] {
        dataPoints.map { $0.close }
    }
    
    var dates: [Date] {
        dataPoints.map { $0.date }
    }
}

struct StockMetrics {
    let ticker: String
    let periodDays: Int
    let periodReturn: Double      // Actual return over the data period
    let annualizedReturn: Double  // CAGR - properly annualized
    let volatility: Double        // Annualized volatility
    let minPrice: Double
    let maxPrice: Double
    let currentPrice: Double
    let startDate: Date
    let endDate: Date
    
    var formattedPeriod: String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        return "\(formatter.string(from: startDate)) to \(formatter.string(from: endDate))"
    }
}

// MARK: - AlphaVantage Service

class AlphaVantageService {
    static let shared = AlphaVantageService()
    
    private let apiKey: String
    private let baseURL = "https://www.alphavantage.co/query"
    private let dateFormatter: DateFormatter
    
    private init() {
        // TODO: Insert API Key here
        self.apiKey =  ""
        
        dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
    }
    
    func setApiKey(_ key: String) {
        // For runtime configuration if needed
    }
    
    // MARK: - Fetch Time Series Data
    
    func fetchTimeSeries(ticker: String, range: String = "1y") async throws -> StockTimeSeries {
        // Determine function based on range
        // Free tier: compact=100 data points for daily, but weekly/monthly return full history
        // Use weekly data for 6m+ to get more complete coverage on free tier
        let function: String
        let useWeekly: Bool
        
        switch range.lowercased() {
        case "1m", "1mo", "1month":
            function = "TIME_SERIES_DAILY"
            useWeekly = false
        case "3m", "3mo":
            function = "TIME_SERIES_DAILY"
            useWeekly = false
        case "6m", "6mo":
            function = "TIME_SERIES_WEEKLY"  // ~26 weeks, better coverage than 100 daily points
            useWeekly = true
        case "1y", "1yr", "12m":
            function = "TIME_SERIES_WEEKLY"  // ~52 weeks for full year coverage
            useWeekly = true
        case "5y", "5yr":
            function = "TIME_SERIES_WEEKLY"  // ~260 weeks
            useWeekly = true
        default:
            function = "TIME_SERIES_DAILY"
            useWeekly = false
        }
        
        var components = URLComponents(string: baseURL)!
        components.queryItems = [
            URLQueryItem(name: "function", value: function),
            URLQueryItem(name: "symbol", value: ticker.uppercased()),
            URLQueryItem(name: "apikey", value: apiKey)
            // Note: Weekly/Monthly endpoints return full history even on free tier
            // Daily with outputsize=compact returns only 100 data points
        ]
        
        print("📡 API Call: \(function) for \(ticker.uppercased()), range: \(range), weekly: \(useWeekly)")
        
        guard let url = components.url else {
            throw AlphaVantageError.invalidURL
        }
        
        let (data, response) = try await URLSession.shared.data(from: url)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 else {
            throw AlphaVantageError.networkError
        }
        
        // Check for API error messages
        if let errorResponse = try? JSONDecoder().decode([String: String].self, from: data),
           let errorMessage = errorResponse["Error Message"] ?? errorResponse["Note"] {
            throw AlphaVantageError.apiError(errorMessage)
        }
        
        let timeSeries = try JSONDecoder().decode(AlphaVantageTimeSeries.self, from: data)
        
        // Filter data based on range
        let calendar = Calendar.current
        let now = Date()
        let startDate: Date
        
        switch range.lowercased() {
        case "1m", "1mo", "1month":
            startDate = calendar.date(byAdding: .month, value: -1, to: now)!
        case "3m", "3mo":
            startDate = calendar.date(byAdding: .month, value: -3, to: now)!
        case "6m", "6mo":
            startDate = calendar.date(byAdding: .month, value: -6, to: now)!
        case "1y", "1yr", "12m":
            startDate = calendar.date(byAdding: .year, value: -1, to: now)!
        case "5y", "5yr":
            startDate = calendar.date(byAdding: .year, value: -5, to: now)!
        default:
            startDate = calendar.date(byAdding: .year, value: -1, to: now)!
        }
        
        let dataPoints = timeSeries.timeSeries.compactMap { (dateString, data) -> StockDataPoint? in
            guard let date = dateFormatter.date(from: dateString),
                  date >= startDate,
                  let close = Double(data.close),
                  let volume = Int(data.volume) else {
                return nil
            }
            return StockDataPoint(date: date, close: close, volume: volume)
        }
        .sorted { $0.date < $1.date }
        
        return StockTimeSeries(ticker: ticker.uppercased(), dataPoints: dataPoints)
    }
    
    // MARK: - Compute Metrics
    
    func computeMetrics(series: StockTimeSeries) -> StockMetrics {
        let prices = series.prices
        let dataPoints = series.dataPoints
        
        guard prices.count > 1, let firstPoint = dataPoints.first, let lastPoint = dataPoints.last else {
            return StockMetrics(
                ticker: series.ticker,
                periodDays: 0,
                periodReturn: 0,
                annualizedReturn: 0,
                volatility: 0,
                minPrice: prices.first ?? 0,
                maxPrice: prices.first ?? 0,
                currentPrice: prices.last ?? 0,
                startDate: Date(),
                endDate: Date()
            )
        }
        
        // Calculate the actual period in days
        let periodDays = Calendar.current.dateComponents([.day], from: firstPoint.date, to: lastPoint.date).day ?? 1
        let tradingDays = prices.count - 1
        
        // Calculate daily returns
        var returns: [Double] = []
        for i in 1..<prices.count {
            let dailyReturn = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(dailyReturn)
        }
        
        // Period return (actual return over the data period)
        let periodReturn = (prices.last! - prices.first!) / prices.first!
        
        // Annualized return using CAGR formula: (1 + total_return)^(365/days) - 1
        let yearsInPeriod = Double(periodDays) / 365.0
        let annualizedReturn: Double
        if yearsInPeriod > 0 && periodReturn > -1 {
            annualizedReturn = pow(1 + periodReturn, 1 / yearsInPeriod) - 1
        } else {
            annualizedReturn = periodReturn
        }
        
        // Volatility (standard deviation of daily returns, annualized)
        let meanDailyReturn = returns.reduce(0, +) / Double(returns.count)
        let variance = returns.map { pow($0 - meanDailyReturn, 2) }.reduce(0, +) / Double(returns.count)
        let dailyVolatility = sqrt(variance)
        let annualizedVolatility = dailyVolatility * sqrt(252)
        
        return StockMetrics(
            ticker: series.ticker,
            periodDays: periodDays,
            periodReturn: periodReturn,
            annualizedReturn: annualizedReturn,
            volatility: annualizedVolatility,
            minPrice: prices.min() ?? 0,
            maxPrice: prices.max() ?? 0,
            currentPrice: prices.last ?? 0,
            startDate: firstPoint.date,
            endDate: lastPoint.date
        )
    }
    
    // MARK: - Correlation
    
    func computeCorrelation(series1: StockTimeSeries, series2: StockTimeSeries) -> Double {
        // Align the series by date
        let dates1 = Set(series1.dataPoints.map { dateFormatter.string(from: $0.date) })
        let dates2 = Set(series2.dataPoints.map { dateFormatter.string(from: $0.date) })
        let commonDates = dates1.intersection(dates2)
        
        let prices1 = series1.dataPoints
            .filter { commonDates.contains(dateFormatter.string(from: $0.date)) }
            .sorted { $0.date < $1.date }
            .map { $0.close }
        
        let prices2 = series2.dataPoints
            .filter { commonDates.contains(dateFormatter.string(from: $0.date)) }
            .sorted { $0.date < $1.date }
            .map { $0.close }
        
        guard prices1.count == prices2.count, prices1.count > 1 else {
            return 0
        }
        
        // Calculate returns
        var returns1: [Double] = []
        var returns2: [Double] = []
        
        for i in 1..<prices1.count {
            returns1.append((prices1[i] - prices1[i-1]) / prices1[i-1])
            returns2.append((prices2[i] - prices2[i-1]) / prices2[i-1])
        }
        
        // Pearson correlation
        let mean1 = returns1.reduce(0, +) / Double(returns1.count)
        let mean2 = returns2.reduce(0, +) / Double(returns2.count)
        
        var numerator: Double = 0
        var denominator1: Double = 0
        var denominator2: Double = 0
        
        for i in 0..<returns1.count {
            let diff1 = returns1[i] - mean1
            let diff2 = returns2[i] - mean2
            numerator += diff1 * diff2
            denominator1 += diff1 * diff1
            denominator2 += diff2 * diff2
        }
        
        let denominator = sqrt(denominator1 * denominator2)
        return denominator > 0 ? numerator / denominator : 0
    }
}

// MARK: - Errors

enum AlphaVantageError: LocalizedError {
    case invalidURL
    case networkError
    case decodingError
    case apiError(String)
    case noData
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError:
            return "Network error occurred"
        case .decodingError:
            return "Failed to decode response"
        case .apiError(let message):
            return "API Error: \(message)"
        case .noData:
            return "No data available"
        }
    }
}

