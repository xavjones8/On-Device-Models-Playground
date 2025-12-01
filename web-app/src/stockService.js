/**
 * AlphaVantage Stock Service
 * 
 * This service handles all stock data fetching and financial metric calculations.
 * It uses the AlphaVantage API (https://www.alphavantage.co/) for historical stock data.
 * 
 * Key Features:
 * - Fetches daily/weekly time series data
 * - Computes financial metrics (returns, volatility, CAGR)
 * - Calculates correlation between stocks
 * - Caches data to minimize API calls (free tier: 5 calls/min, 500/day)
 * 
 * Note: The free tier returns only 100 data points for daily data,
 * so we use weekly data for 6+ month ranges to get better coverage.
 */

// API configuration - uses Vite's environment variable system
// Create a .env file with VITE_ALPHAVANTAGE_API_KEY=your_key
const API_KEY = import.meta.env.VITE_ALPHAVANTAGE_API_KEY || '';
const BASE_URL = 'https://www.alphavantage.co/query';

// In-memory cache to store fetched series data
// Key: ticker symbol (uppercase), Value: StockTimeSeries object
const seriesCache = {};

/**
 * Fetches historical stock price data from AlphaVantage
 * 
 * @param {string} ticker - Stock ticker symbol (e.g., 'AAPL', 'NVDA')
 * @param {string} range - Time range: '1m', '3m', '6m', '1y', '5y'
 * @returns {Promise<Object>} StockTimeSeries with ticker, dataPoints array, and prices array
 * @throws {Error} If API call fails or returns an error
 * 
 * API Behavior:
 * - Daily data (1m, 3m): Returns ~100 most recent trading days
 * - Weekly data (6m+): Returns full history, filtered to requested range
 */
export async function fetchTimeSeries(ticker, range = '1y') {
  const upperTicker = ticker.toUpperCase().replace('$', '');
  
  // Determine which API endpoint to use based on requested range
  // Weekly data provides better coverage for longer time periods on free tier
  let func;
  let useWeekly = false;
  
  switch (range.toLowerCase()) {
    case '1m':
    case '3m':
      func = 'TIME_SERIES_DAILY';
      break;
    case '6m':
    case '1y':
    case '5y':
      func = 'TIME_SERIES_WEEKLY';
      useWeekly = true;
      break;
    default:
      func = 'TIME_SERIES_DAILY';
  }
  
  const url = `${BASE_URL}?function=${func}&symbol=${upperTicker}&apikey=${API_KEY}`;
  
  console.log(`📡 Fetching ${func} for ${upperTicker}, range: ${range}`);
  
  try {
    const response = await fetch(url);
    const data = await response.json();
    
    // Check for API error responses
    if (data['Error Message']) {
      throw new Error(data['Error Message']);
    }
    if (data['Note']) {
      throw new Error('API rate limit exceeded. Please wait a minute.');
    }
    
    // Parse time series data - key name differs between daily/weekly endpoints
    const timeSeriesKey = useWeekly ? 'Weekly Time Series' : 'Time Series (Daily)';
    const timeSeries = data[timeSeriesKey];
    
    if (!timeSeries) {
      throw new Error(`No data found for ${upperTicker}`);
    }
    
    // Convert API response to array of data points
    // AlphaVantage returns data as { "YYYY-MM-DD": { "1. open": "...", ... } }
    const dataPoints = Object.entries(timeSeries)
      .map(([dateStr, values]) => ({
        date: new Date(dateStr),
        close: parseFloat(values['4. close']),
        volume: parseInt(values['5. volume'] || values['6. volume'] || 0)
      }))
      .sort((a, b) => a.date - b.date); // Sort chronologically (oldest first)
    
    // Filter to requested date range
    const now = new Date();
    let startDate;
    switch (range.toLowerCase()) {
      case '1m': startDate = new Date(now.setMonth(now.getMonth() - 1)); break;
      case '3m': startDate = new Date(now.setMonth(now.getMonth() - 3)); break;
      case '6m': startDate = new Date(now.setMonth(now.getMonth() - 6)); break;
      case '1y': startDate = new Date(now.setFullYear(now.getFullYear() - 1)); break;
      case '5y': startDate = new Date(now.setFullYear(now.getFullYear() - 5)); break;
      default: startDate = new Date(now.setFullYear(now.getFullYear() - 1));
    }
    
    const filteredPoints = dataPoints.filter(p => p.date >= startDate);
    
    // Build the series object
    const series = {
      ticker: upperTicker,
      dataPoints: filteredPoints,
      prices: filteredPoints.map(p => p.close)
    };
    
    // Cache for use by other functions (metrics, correlation, charts)
    seriesCache[upperTicker] = series;
    
    return series;
  } catch (error) {
    console.error(`Failed to fetch ${upperTicker}:`, error);
    throw error;
  }
}

/**
 * Computes financial metrics for a stock time series
 * 
 * @param {Object} series - StockTimeSeries object from fetchTimeSeries
 * @returns {Object} StockMetrics with:
 *   - ticker: Stock symbol
 *   - periodDays: Number of data points
 *   - periodReturn: Total return over the period (decimal, e.g., 0.15 = 15%)
 *   - annualizedReturn: CAGR (Compound Annual Growth Rate)
 *   - volatility: Annualized standard deviation of returns
 *   - currentPrice: Most recent closing price
 *   - minPrice/maxPrice: Price range over the period
 *   - startDate/endDate: Date range of the data
 * 
 * Financial Formulas:
 * - Period Return: (endPrice - startPrice) / startPrice
 * - CAGR: (1 + periodReturn)^(365/days) - 1
 * - Volatility: stdDev(dailyReturns) * sqrt(252)  // 252 trading days/year
 */
export function computeMetrics(series) {
  const prices = series.prices;
  
  // Handle edge case of insufficient data
  if (prices.length < 2) {
    return {
      ticker: series.ticker,
      periodDays: 0,
      periodReturn: 0,
      annualizedReturn: 0,
      volatility: 0,
      currentPrice: prices[0] || 0,
      minPrice: prices[0] || 0,
      maxPrice: prices[0] || 0,
      startDate: null,
      endDate: null
    };
  }
  
  const periodDays = series.dataPoints.length;
  const firstPrice = prices[0];
  const lastPrice = prices[prices.length - 1];
  
  // Calculate period return (simple return)
  const periodReturn = (lastPrice - firstPrice) / firstPrice;
  
  // Calculate annualized return using CAGR formula
  // CAGR = (1 + total_return)^(365/days) - 1
  let annualizedReturn = 0;
  if (periodDays > 0 && periodReturn > -1) {
    annualizedReturn = Math.pow(1 + periodReturn, 365 / periodDays) - 1;
  }
  
  // Calculate daily/weekly returns for volatility
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i-1]) / prices[i-1]);
  }
  
  // Calculate volatility (annualized standard deviation)
  // Formula: stdDev * sqrt(trading_periods_per_year)
  // Using 252 for daily data (trading days), adjusted for weekly
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
  const periodVolatility = Math.sqrt(variance);
  const annualizedVolatility = periodVolatility * Math.sqrt(252); // Approximate annualization
  
  return {
    ticker: series.ticker,
    periodDays,
    periodReturn,
    annualizedReturn,
    volatility: annualizedVolatility,
    currentPrice: lastPrice,
    minPrice: Math.min(...prices),
    maxPrice: Math.max(...prices),
    startDate: series.dataPoints[0]?.date,
    endDate: series.dataPoints[series.dataPoints.length - 1]?.date
  };
}

/**
 * Computes Pearson correlation coefficient between two stock series
 * 
 * @param {Object} series1 - First StockTimeSeries
 * @param {Object} series2 - Second StockTimeSeries
 * @returns {number} Correlation coefficient between -1 and 1
 *   - +1: Perfect positive correlation (move together)
 *   - 0: No correlation
 *   - -1: Perfect negative correlation (move opposite)
 * 
 * Interpretation:
 * - > 0.7: Highly correlated (poor diversification)
 * - 0.3 to 0.7: Moderately correlated
 * - < 0.3: Weakly correlated (good diversification)
 * - < 0: Negatively correlated (hedging potential)
 * 
 * Note: Only uses dates where both stocks have data
 */
export function computeCorrelation(series1, series2) {
  // Find common dates between both series
  const dates1 = new Set(series1.dataPoints.map(p => p.date.toISOString().split('T')[0]));
  const dates2 = new Set(series2.dataPoints.map(p => p.date.toISOString().split('T')[0]));
  
  const commonDates = [...dates1].filter(d => dates2.has(d));
  
  // Extract prices for common dates only
  const prices1 = series1.dataPoints
    .filter(p => commonDates.includes(p.date.toISOString().split('T')[0]))
    .sort((a, b) => a.date - b.date)
    .map(p => p.close);
    
  const prices2 = series2.dataPoints
    .filter(p => commonDates.includes(p.date.toISOString().split('T')[0]))
    .sort((a, b) => a.date - b.date)
    .map(p => p.close);
  
  if (prices1.length !== prices2.length || prices1.length < 2) {
    return 0;
  }
  
  // Calculate Pearson correlation coefficient
  // Formula: Σ[(xi - x̄)(yi - ȳ)] / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)
  const mean1 = prices1.reduce((a, b) => a + b, 0) / prices1.length;
  const mean2 = prices2.reduce((a, b) => a + b, 0) / prices2.length;
  
  let numerator = 0;
  let denom1 = 0;
  let denom2 = 0;
  
  for (let i = 0; i < prices1.length; i++) {
    const diff1 = prices1[i] - mean1;
    const diff2 = prices2[i] - mean2;
    numerator += diff1 * diff2;
    denom1 += diff1 * diff1;
    denom2 += diff2 * diff2;
  }
  
  const denominator = Math.sqrt(denom1 * denom2);
  return denominator === 0 ? 0 : numerator / denominator;
}

/**
 * Retrieves a previously fetched series from cache
 * @param {string} ticker - Stock ticker symbol
 * @returns {Object|undefined} Cached StockTimeSeries or undefined
 */
export function getCachedSeries(ticker) {
  return seriesCache[ticker.toUpperCase().replace('$', '')];
}

/**
 * Clears all cached series data
 * Call this when starting a new analysis session
 */
export function clearCache() {
  Object.keys(seriesCache).forEach(key => delete seriesCache[key]);
}

// ============================================
// Formatting Utilities
// ============================================

/**
 * Formats a Date object for display
 * @param {Date} date - Date to format
 * @returns {string} Formatted date string (e.g., "Jan 15, 2024")
 */
export function formatDate(date) {
  if (!date) return 'N/A';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

/**
 * Formats a decimal value as percentage
 * @param {number} value - Decimal value (e.g., 0.15 for 15%)
 * @param {boolean} includeSign - Whether to include + for positive values
 * @returns {string} Formatted percentage (e.g., "+15.00%")
 */
export function formatPercent(value, includeSign = true) {
  const pct = (value * 100).toFixed(2);
  if (includeSign && value >= 0) return `+${pct}%`;
  return `${pct}%`;
}

/**
 * Formats a number as USD currency
 * @param {number} value - Dollar amount
 * @returns {string} Formatted currency (e.g., "$123.45")
 */
export function formatCurrency(value) {
  return `$${value.toFixed(2)}`;
}
