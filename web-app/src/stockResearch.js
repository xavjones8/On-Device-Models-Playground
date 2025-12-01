/**
 * Stock Research Agent
 * 
 * This module provides an agent-like interface for stock analysis.
 * It parses natural language queries, extracts stock tickers and parameters,
 * orchestrates API calls, and generates comprehensive analysis reports.
 * 
 * Architecture:
 * - Natural Language Understanding: Extracts tickers, time ranges, chart types
 * - Tool Orchestration: Calls stockService functions in sequence
 * - Report Generation: Creates markdown-formatted analysis with insights
 * - Optional AI Enhancement: Uses WebLLM for additional investment insights
 * 
 * This is a simplified "agent" pattern that doesn't use true LLM tool-calling,
 * but instead uses pattern matching to understand user intent.
 */

import { llm } from './llm';
import * as stockService from './stockService';

// ============================================
// State Management
// ============================================

// Log of tool calls for UI display (shows what the agent did)
let toolCallLog = [];

// Most recent chart data for rendering
let lastChartData = null;

/**
 * Gets the tool call log for UI display
 * @returns {Array} Array of tool call records
 */
export function getToolCallLog() {
  return toolCallLog;
}

/**
 * Gets the last generated chart data
 * @returns {Object|null} Chart data object or null
 */
export function getLastChartData() {
  return lastChartData;
}

/**
 * Clears all research state for a new session
 */
export function clearResearchState() {
  toolCallLog = [];
  lastChartData = null;
  stockService.clearCache();
}

// ============================================
// Natural Language Understanding
// ============================================

/**
 * Extracts stock tickers from a natural language message
 * 
 * Uses two strategies:
 * 1. Pattern matching for uppercase ticker symbols (AAPL, NVDA, etc.)
 * 2. Company name lookup (Apple → AAPL, Microsoft → MSFT, etc.)
 * 
 * @param {string} message - User's natural language query
 * @returns {string[]} Array of unique ticker symbols
 */
function extractTickers(message) {
  const tickerPattern = /\b([A-Z]{1,5})\b/g;
  const commonTickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'AMD', 'TSLA', 'NFLX', 'IBM', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'SQ', 'SHOP', 'UBER', 'LYFT', 'ABNB', 'COIN', 'HOOD', 'PLTR', 'SNOW', 'NET', 'DDOG', 'ZS', 'CRWD', 'OKTA'];
  
  // Also check for written names
  const nameToTicker = {
    'apple': 'AAPL',
    'microsoft': 'MSFT',
    'google': 'GOOGL',
    'alphabet': 'GOOGL',
    'amazon': 'AMZN',
    'meta': 'META',
    'facebook': 'META',
    'nvidia': 'NVDA',
    'amd': 'AMD',
    'tesla': 'TSLA',
    'netflix': 'NFLX',
    'ibm': 'IBM',
    'intel': 'INTC',
    'oracle': 'ORCL',
    'salesforce': 'CRM',
    'adobe': 'ADBE',
    'paypal': 'PYPL'
  };
  
  const tickers = new Set();
  
  // Check for ticker symbols
  const matches = message.toUpperCase().match(tickerPattern) || [];
  matches.forEach(m => {
    if (commonTickers.includes(m)) {
      tickers.add(m);
    }
  });
  
  // Check for company names
  const lowerMessage = message.toLowerCase();
  Object.entries(nameToTicker).forEach(([name, ticker]) => {
    if (lowerMessage.includes(name)) {
      tickers.add(ticker);
    }
  });
  
  return Array.from(tickers);
}

// Parse time range from message
function extractRange(message) {
  const lower = message.toLowerCase();
  if (lower.includes('5 year') || lower.includes('5y')) return '5y';
  if (lower.includes('1 year') || lower.includes('year') || lower.includes('1y') || lower.includes('12 month')) return '1y';
  if (lower.includes('6 month') || lower.includes('6m')) return '6m';
  if (lower.includes('3 month') || lower.includes('3m')) return '3m';
  if (lower.includes('1 month') || lower.includes('1m')) return '1m';
  return '1y'; // Default
}

// Determine chart type from message
function extractChartType(message) {
  const lower = message.toLowerCase();
  if (lower.includes('performance') || lower.includes('compare') || lower.includes('comparison') || lower.includes('%') || lower.includes('percent')) {
    return 'performance';
  }
  return 'price';
}

// Log a tool call
function logTool(toolName, args, result) {
  const entry = {
    id: Date.now(),
    toolName,
    args: JSON.stringify(args),
    result: result.substring(0, 100) + (result.length > 100 ? '...' : ''),
    timestamp: new Date()
  };
  toolCallLog.push(entry);
  console.log(`🔧 ${toolName}(${entry.args}) => ${entry.result}`);
  return entry;
}

// Execute stock research based on user query
export async function executeResearch(userMessage, onProgress) {
  toolCallLog = [];
  lastChartData = null;
  
  const tickers = extractTickers(userMessage);
  const range = extractRange(userMessage);
  const chartType = extractChartType(userMessage);
  
  if (tickers.length === 0) {
    return {
      response: "I couldn't identify any stock tickers in your message. Please mention specific stocks like AAPL, NVDA, or company names like Apple, Microsoft.",
      chartData: null,
      toolCalls: toolCallLog
    };
  }
  
  onProgress?.({ stage: 'fetching', message: `Fetching data for ${tickers.join(', ')}...` });
  
  // Step 1: Fetch time series for each ticker
  const seriesData = {};
  const metricsData = {};
  
  for (const ticker of tickers) {
    try {
      onProgress?.({ stage: 'fetching', message: `Fetching ${ticker}...` });
      const series = await stockService.fetchTimeSeries(ticker, range);
      seriesData[ticker] = series;
      logTool('fetchTimeSeries', { ticker, range }, `Fetched ${series.dataPoints.length} data points`);
      
      // Compute metrics
      const metrics = stockService.computeMetrics(series);
      metricsData[ticker] = metrics;
      logTool('computeMetrics', { ticker }, `Return: ${stockService.formatPercent(metrics.periodReturn)}`);
    } catch (error) {
      logTool('fetchTimeSeries', { ticker, range }, `Error: ${error.message}`);
    }
  }
  
  if (Object.keys(seriesData).length === 0) {
    return {
      response: "Failed to fetch stock data. The API might be rate limited. Please try again in a minute.",
      chartData: null,
      toolCalls: toolCallLog
    };
  }
  
  onProgress?.({ stage: 'analyzing', message: 'Analyzing data...' });
  
  // Log comparison if multiple stocks
  if (Object.keys(metricsData).length >= 2) {
    const tickerList = Object.keys(metricsData);
    const sorted = tickerList.sort((a, b) => metricsData[b].periodReturn - metricsData[a].periodReturn);
    logTool('compareStocks', { tickers: tickerList }, `Best: ${sorted[0]}`);
  }
  
  // Step 3: Generate chart data
  onProgress?.({ stage: 'charting', message: 'Generating chart...' });
  
  const chartSeries = {};
  const normalizedSeries = {};
  
  Object.entries(seriesData).forEach(([ticker, series]) => {
    chartSeries[ticker] = series.dataPoints.map(p => ({
      date: p.date,
      price: p.close
    }));
    
    // Normalized (% change from start)
    const firstPrice = series.prices[0];
    normalizedSeries[ticker] = series.dataPoints.map(p => ({
      date: p.date,
      value: ((p.close - firstPrice) / firstPrice) * 100
    }));
  });
  
  lastChartData = {
    tickers: Object.keys(seriesData),
    chartType: tickers.length > 1 ? 'performance' : chartType,
    series: chartSeries,
    normalizedSeries
  };
  
  logTool('generateChartData', { tickers: Object.keys(seriesData), chartType: lastChartData.chartType }, 'Chart ready');
  
  // Step 4: Build analysis text with insights
  let analysisText = `## Stock Analysis\n\n`;
  
  Object.entries(metricsData).forEach(([ticker, metrics]) => {
    // Performance interpretation
    const returnDesc = metrics.periodReturn > 0.3 ? 'exceptional' : 
                       metrics.periodReturn > 0.15 ? 'strong' :
                       metrics.periodReturn > 0.05 ? 'moderate' :
                       metrics.periodReturn > -0.05 ? 'flat' :
                       metrics.periodReturn > -0.15 ? 'weak' : 'poor';
    
    // Volatility interpretation
    const volDesc = metrics.volatility > 0.5 ? 'very high' :
                    metrics.volatility > 0.35 ? 'high' :
                    metrics.volatility > 0.2 ? 'moderate' : 'low';
    
    // Risk assessment
    const riskLevel = metrics.volatility > 0.4 ? 'high-risk' :
                      metrics.volatility > 0.25 ? 'moderate-risk' : 'lower-risk';
    
    analysisText += `### ${ticker}\n\n`;
    
    // Summary sentence
    analysisText += `**${ticker}** has shown **${returnDesc} performance** over the analyzed period with a ${stockService.formatPercent(metrics.periodReturn)} return. `;
    analysisText += `The stock exhibits **${volDesc} volatility** (${stockService.formatPercent(metrics.volatility, false)} annualized), making it a ${riskLevel} investment.\n\n`;
    
    // Key metrics
    analysisText += `**Key Metrics:**\n`;
    analysisText += `- Current Price: ${stockService.formatCurrency(metrics.currentPrice)}\n`;
    analysisText += `- Period Return: ${stockService.formatPercent(metrics.periodReturn)}\n`;
    analysisText += `- Annualized Return (CAGR): ${stockService.formatPercent(metrics.annualizedReturn)}\n`;
    analysisText += `- Annualized Volatility: ${stockService.formatPercent(metrics.volatility, false)}\n`;
    analysisText += `- Price Range: ${stockService.formatCurrency(metrics.minPrice)} - ${stockService.formatCurrency(metrics.maxPrice)}\n\n`;
    
    // Price movement insight
    const priceRange = metrics.maxPrice - metrics.minPrice;
    const priceSwing = (priceRange / metrics.minPrice) * 100;
    if (priceSwing > 50) {
      analysisText += `⚠️ The stock experienced significant price swings (${priceSwing.toFixed(0)}% range), indicating substantial volatility during this period.\n\n`;
    } else if (priceSwing > 25) {
      analysisText += `📊 The stock showed notable price movement (${priceSwing.toFixed(0)}% range) during this period.\n\n`;
    }
  });
  
  // Enhanced comparison text
  if (Object.keys(metricsData).length >= 2) {
    const tickerList = Object.keys(metricsData);
    const sorted = tickerList.sort((a, b) => metricsData[b].periodReturn - metricsData[a].periodReturn);
    const best = sorted[0];
    const worst = sorted[sorted.length - 1];
    const bestMetrics = metricsData[best];
    const worstMetrics = metricsData[worst];
    
    const returnDiff = (bestMetrics.periodReturn - worstMetrics.periodReturn) * 100;
    
    analysisText += `## Comparison Analysis\n\n`;
    
    // Performance ranking
    analysisText += `**Performance Ranking:**\n`;
    sorted.forEach((ticker, i) => {
      const emoji = i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉';
      analysisText += `${emoji} **${ticker}**: ${stockService.formatPercent(metricsData[ticker].periodReturn)}\n`;
    });
    analysisText += `\n`;
    
    // Comparison insight
    if (returnDiff > 20) {
      analysisText += `**${best} is significantly outperforming ${worst}** with a ${returnDiff.toFixed(1)} percentage point advantage. `;
    } else if (returnDiff > 10) {
      analysisText += `**${best} is outperforming ${worst}** with a ${returnDiff.toFixed(1)} percentage point lead. `;
    } else if (returnDiff > 5) {
      analysisText += `**${best} has a modest edge over ${worst}** (${returnDiff.toFixed(1)} percentage points). `;
    } else {
      analysisText += `**${best} and ${worst} are performing similarly**, with only a ${returnDiff.toFixed(1)} percentage point difference. `;
    }
    
    // Risk comparison
    const bestVol = bestMetrics.volatility;
    const worstVol = worstMetrics.volatility;
    if (bestVol < worstVol * 0.7) {
      analysisText += `Notably, ${best} achieves this with lower volatility, suggesting better risk-adjusted returns.\n\n`;
    } else if (bestVol > worstVol * 1.3) {
      analysisText += `However, ${best}'s higher volatility means investors took on more risk for these returns.\n\n`;
    } else {
      analysisText += `Both stocks show similar volatility profiles.\n\n`;
    }
    
    // Correlation insight
    if (tickerList.length === 2) {
      const corr = stockService.computeCorrelation(seriesData[tickerList[0]], seriesData[tickerList[1]]);
      analysisText += `**Correlation:** ${corr.toFixed(3)}\n`;
      if (corr > 0.7) {
        analysisText += `These stocks are highly correlated and tend to move together. Holding both may not provide much diversification benefit.\n`;
      } else if (corr > 0.3) {
        analysisText += `These stocks show moderate correlation. They share some market factors but also have independent drivers.\n`;
      } else if (corr > -0.3) {
        analysisText += `These stocks have low correlation, making them good candidates for portfolio diversification.\n`;
      } else {
        analysisText += `These stocks are negatively correlated - when one goes up, the other tends to go down. This can provide strong hedging benefits.\n`;
      }
      logTool('computeCorrelation', { tickers: tickerList }, `Correlation: ${corr.toFixed(3)}`);
    }
  } else {
    // Single stock recommendation
    const ticker = Object.keys(metricsData)[0];
    const metrics = metricsData[ticker];
    
    analysisText += `## Investment Consideration\n\n`;
    
    if (metrics.periodReturn > 0.15 && metrics.volatility < 0.35) {
      analysisText += `${ticker} shows strong returns with manageable volatility - a favorable risk/reward profile for growth-oriented investors.\n`;
    } else if (metrics.periodReturn > 0.15 && metrics.volatility >= 0.35) {
      analysisText += `${ticker} has delivered strong returns but with high volatility. Suitable for investors with higher risk tolerance.\n`;
    } else if (metrics.periodReturn > 0 && metrics.volatility < 0.25) {
      analysisText += `${ticker} offers steady, positive returns with lower volatility - suitable for more conservative investors.\n`;
    } else if (metrics.periodReturn < 0) {
      analysisText += `${ticker} has underperformed during this period. Consider waiting for signs of reversal or researching fundamental factors.\n`;
    } else {
      analysisText += `${ticker} shows mixed signals. Consider your investment timeline and risk tolerance before making decisions.\n`;
    }
  }
  
  // Step 5: Generate AI insights if LLM is loaded
  let aiInsights = '';
  if (llm.isLoaded) {
    onProgress?.({ stage: 'insights', message: 'Generating AI insights...' });
    try {
      const tickers = Object.keys(metricsData);
      const metricsStr = tickers.map(t => {
        const m = metricsData[t];
        return `${t}: Return ${(m.periodReturn * 100).toFixed(1)}%, Volatility ${(m.volatility * 100).toFixed(1)}%, Price $${m.currentPrice.toFixed(2)}`;
      }).join('; ');
      
      const insightPrompt = `You are a financial analyst. Given this stock data: ${metricsStr}

Provide a brief 2-3 sentence actionable investment insight. Focus on:
- Which stock looks more attractive and why
- Key risk factors to consider
- A clear recommendation

Be direct and specific.`;
      
      aiInsights = await llm.generate(insightPrompt, null);
      aiInsights = `\n\n---\n\n## 🤖 AI Investment Insight\n\n${aiInsights}`;
    } catch (e) {
      console.warn('Failed to generate AI insights:', e);
    }
  }
  
  return {
    response: analysisText + aiInsights,
    chartData: lastChartData,
    toolCalls: toolCallLog
  };
}

