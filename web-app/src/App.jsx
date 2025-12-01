import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { classifier } from './classifier';
import { llm, getAvailableModels } from './llm';
import { executeResearch, clearResearchState, getLastChartData } from './stockResearch';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('analyzer');
  const [selectedModel, setSelectedModel] = useState(null);
  const [availableModels, setAvailableModels] = useState([]);
  const [llmLoaded, setLlmLoaded] = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmProgress, setLlmProgress] = useState({ text: '', progress: 0 });

  useEffect(() => {
    const models = getAvailableModels();
    setAvailableModels(models);
    // Default to smallest model and auto-load it
    if (models.length > 0) {
      const defaultModel = models[0].id;
      setSelectedModel(defaultModel);
      // Auto-load the default model
      loadLLMAsync(defaultModel);
    }
  }, []);

  const loadLLMAsync = async (modelId) => {
    if (!modelId || llmLoading) return;
    
    setLlmLoading(true);
    setLlmLoaded(false);
    setLlmProgress({ text: 'Starting...', progress: 0 });
    
    try {
      await llm.load(modelId, (progress) => {
        setLlmProgress(progress);
      });
      setLlmLoaded(true);
    } catch (err) {
      console.error('Failed to load LLM:', err);
      setLlmProgress({ text: `Error: ${err.message}`, progress: 0 });
    } finally {
      setLlmLoading(false);
    }
  };

  const loadLLM = async (modelId) => {
    if (!modelId || llmLoading) return;
    
    setLlmLoading(true);
    setLlmLoaded(false);
    setLlmProgress({ text: 'Starting...', progress: 0 });
    
    try {
      await llm.load(modelId, (progress) => {
        setLlmProgress(progress);
      });
      setLlmLoaded(true);
    } catch (err) {
      console.error('Failed to load LLM:', err);
      setLlmProgress({ text: `Error: ${err.message}`, progress: 0 });
    } finally {
      setLlmLoading(false);
    }
  };

  const handleModelChange = (modelId) => {
    setSelectedModel(modelId);
    setLlmLoaded(false);
    setLlmProgress({ text: '', progress: 0 });
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Prompt Analyzer</h1>
        <p className="subtitle">On-device AI • Powered by ONNX & WebLLM</p>
      </header>

      <nav className="tabs">
        <button 
          className={`tab ${activeTab === 'analyzer' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyzer')}
        >
          ✨ Analyzer
        </button>
        <button 
          className={`tab ${activeTab === 'research' ? 'active' : ''}`}
          onClick={() => setActiveTab('research')}
        >
          📈 Research
        </button>
        <button 
          className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => setActiveTab('chat')}
        >
          💬 Chat
        </button>
      </nav>

      <main className="main">
        {activeTab === 'analyzer' && (
          <AnalyzerTab 
            selectedModel={selectedModel}
            availableModels={availableModels}
            onModelChange={handleModelChange}
            llmLoaded={llmLoaded}
            llmLoading={llmLoading}
            llmProgress={llmProgress}
            loadLLM={loadLLM}
          />
        )}
        {activeTab === 'research' && (
          <ResearchTab 
            selectedModel={selectedModel}
            availableModels={availableModels}
            onModelChange={handleModelChange}
            llmLoaded={llmLoaded}
            llmLoading={llmLoading}
            llmProgress={llmProgress}
            loadLLM={loadLLM}
          />
        )}
        {activeTab === 'chat' && (
          <ChatTab 
            selectedModel={selectedModel}
            availableModels={availableModels}
            onModelChange={handleModelChange}
            llmLoaded={llmLoaded}
            llmLoading={llmLoading}
            llmProgress={llmProgress}
            loadLLM={loadLLM}
          />
        )}
      </main>

      <footer className="footer">
        <p>🔒 Running entirely in your browser • No data sent to servers</p>
      </footer>
    </div>
  );
}

function ModelSelector({ selectedModel, availableModels, onModelChange, llmLoaded, llmLoading, llmProgress, loadLLM }) {
  const groupedModels = availableModels.reduce((acc, model) => {
    if (!acc[model.family]) acc[model.family] = [];
    acc[model.family].push(model);
    return acc;
  }, {});

  const sizeLabels = {
    'XS': '< 500MB',
    'S': '~1GB',
    'M': '~2GB', 
    'L': '~4GB',
    'XL': '> 4GB'
  };

  return (
    <div className="model-selector card">
      <div className="model-header">
        <span>🤖</span>
        <h3>AI Model</h3>
        {llmLoaded && <span className="model-status loaded">● Loaded</span>}
        {llmLoading && <span className="model-status loading">● Loading...</span>}
      </div>
      
      <div className="model-select-row">
        <select 
          value={selectedModel || ''} 
          onChange={(e) => onModelChange(e.target.value)}
          disabled={llmLoading}
        >
          {Object.entries(groupedModels).map(([family, models]) => (
            <optgroup key={family} label={family}>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name} [{sizeLabels[model.size] || model.size}]
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        
        <button 
          className="load-model-btn"
          onClick={() => loadLLM(selectedModel)}
          disabled={llmLoading || llmLoaded}
        >
          {llmLoading ? 'Loading...' : llmLoaded ? '✓ Ready' : 'Load Model'}
        </button>
      </div>

      {llmLoading && llmProgress.progress > 0 && (
        <div className="model-progress">
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${llmProgress.progress * 100}%` }} />
          </div>
          <span className="progress-text">{llmProgress.text}</span>
        </div>
      )}
    </div>
  );
}

function AnalyzerTab({ selectedModel, availableModels, onModelChange, llmLoaded, llmLoading, llmProgress, loadLLM }) {
  const [classifierLoading, setClassifierLoading] = useState(true);
  const [classifierStatus, setClassifierStatus] = useState('Initializing...');
  const [classifierProgress, setClassifierProgress] = useState({ loaded: 0, total: 702, cached: false });
  const [prompt, setPrompt] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [aiResponse, setAiResponse] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [taskExpanded, setTaskExpanded] = useState(false);
  const [complexityExpanded, setComplexityExpanded] = useState(false);

  useEffect(() => {
    const loadClassifier = async () => {
      try {
        await classifier.load(
          (progress) => setClassifierProgress(progress),
          (status) => setClassifierStatus(status)
        );
        setClassifierLoading(false);
      } catch (err) {
        setError(`Failed to load classifier: ${err.message}`);
        setClassifierLoading(false);
      }
    };
    loadClassifier();
  }, []);

  const analyzePrompt = async () => {
    if (!prompt.trim()) return;
    
    setIsAnalyzing(true);
    setAiResponse('');
    setError(null);
    
    try {
      // Run classification
      const result = await classifier.predict(prompt);
      setResults(result);
      setIsAnalyzing(false);
      
      // Generate AI response if model loaded
      if (llmLoaded) {
        setIsGenerating(true);
        await llm.generate(prompt, (text) => setAiResponse(text));
        setIsGenerating(false);
      }
    } catch (err) {
      setError(`Analysis failed: ${err.message}`);
      setIsAnalyzing(false);
      setIsGenerating(false);
    }
  };

  const getScoreColor = (score) => {
    if (score > 0.7) return '#ef4444';
    if (score > 0.5) return '#f97316';
    if (score > 0.3) return '#eab308';
    return '#22c55e';
  };

  const getAggregateColor = (score) => {
    if (score > 0.7) return '#ef4444';
    if (score > 0.4) return '#f97316';
    return '#22c55e';
  };

  const filteredTasks = results?.taskType?.allProbabilities
    ? Object.entries(results.taskType.allProbabilities)
        .filter(([_, prob]) => prob >= 0.1)
        .sort((a, b) => b[1] - a[1])
    : [];

  const complexityItems = results ? [
    { label: 'Creativity', value: results.complexity.creativity, icon: '🎨' },
    { label: 'Reasoning', value: results.complexity.reasoning, icon: '🧠' },
    { label: 'Constraints', value: results.complexity.constraint, icon: '📋' },
    { label: 'Domain Knowledge', value: results.complexity.domain, icon: '📚' },
    { label: 'Context Required', value: results.complexity.contextual, icon: '📄' },
    { label: 'Few-Shot Examples', value: results.complexity.fewShots, icon: '📝' },
  ] : [];

  const progressPercent = classifierProgress.total > 0 
    ? Math.min((classifierProgress.loaded / classifierProgress.total) * 100, 100)
    : 0;

  const copyResponse = () => {
    navigator.clipboard.writeText(aiResponse);
  };

  if (classifierLoading) {
    return (
      <div className="card loading-card">
        <div className="spinner"></div>
        <p className="loading-status">{classifierStatus}</p>
        
        {classifierStatus.includes('ONNX') && (
          <div className="progress-container">
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: `${progressPercent}%` }} />
            </div>
            <div className="progress-info">
              {classifierProgress.cached ? (
                <span className="cached-badge">✓ Cached</span>
              ) : (
                <span>{classifierProgress.loaded.toFixed(1)} / {classifierProgress.total.toFixed(0)} MB</span>
              )}
              <span>{progressPercent.toFixed(0)}%</span>
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <>
      {/* Model Selector */}
      <ModelSelector 
        selectedModel={selectedModel}
        availableModels={availableModels}
        onModelChange={onModelChange}
        llmLoaded={llmLoaded}
        llmLoading={llmLoading}
        llmProgress={llmProgress}
        loadLLM={loadLLM}
      />

      {/* Input Section */}
      <div className="card input-card">
        <label htmlFor="prompt">Enter your prompt</label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Write a Python function that implements quicksort algorithm with detailed comments..."
          rows={4}
        />
        <button 
          className="analyze-btn"
          onClick={analyzePrompt}
          disabled={!prompt.trim() || isAnalyzing || isGenerating}
        >
          {isAnalyzing || isGenerating ? (
            <>
              <span className="btn-spinner"></span>
              {isAnalyzing ? 'Classifying...' : 'Generating...'}
            </>
          ) : (
            <>
              <span>✨</span>
              Analyze Prompt
            </>
          )}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {/* AI Response */}
      {(aiResponse || isGenerating) && (
        <div className="card ai-response-card">
          <div className="ai-header">
            <span>🤖</span>
            <h3>AI Response</h3>
            {aiResponse && (
              <button className="copy-btn" onClick={copyResponse} title="Copy">
                📋
              </button>
            )}
          </div>
          <div className="ai-content">
            {isGenerating && !aiResponse ? (
              <div className="generating">
                <span className="typing-indicator">●●●</span>
                Thinking...
              </div>
            ) : (
              <p className="ai-text">{aiResponse}</p>
            )}
          </div>
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="results">
          {/* Task Type Card */}
          <div className="card results-card">
            <div className="card-header" onClick={() => setTaskExpanded(!taskExpanded)}>
              <div className="header-left">
                <span className="card-icon">🏷️</span>
                <h2>Task Classification</h2>
              </div>
              <div className="header-right">
                <div className="primary-preview">
                  <span className="preview-task">{results.taskType.taskType1}</span>
                  <span className="preview-confidence">
                    {Math.round(results.taskType.probability * 100)}%
                  </span>
                </div>
                <span className={`chevron ${taskExpanded ? 'expanded' : ''}`}>▼</span>
              </div>
            </div>

            {taskExpanded && (
              <div className="card-content">
                <div className="primary-result">
                  <div className="result-item">
                    <span className="result-label">Primary Task</span>
                    <span className="result-value task-name">{results.taskType.taskType1}</span>
                  </div>
                  <div className="result-item">
                    <span className="result-label">Confidence</span>
                    <span className="result-value confidence">
                      {Math.round(results.taskType.probability * 100)}%
                    </span>
                  </div>
                </div>

                {results.taskType.taskType2 && (
                  <div className="secondary-task">
                    <span className="secondary-label">Also likely:</span>
                    <span className="secondary-value">{results.taskType.taskType2}</span>
                  </div>
                )}

                {filteredTasks.length > 0 && (
                  <div className="task-chart">
                    <h3>All Categories (&gt;10%)</h3>
                    {filteredTasks.map(([name, prob]) => (
                      <div key={name} className={`task-bar ${name === results.taskType.taskType1 ? 'top' : ''}`}>
                        <span className="bar-label">{name}</span>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
                        </div>
                        <span className="bar-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Complexity Card */}
          <div className="card results-card">
            <div className="card-header" onClick={() => setComplexityExpanded(!complexityExpanded)}>
              <div className="header-left">
                <span className="card-icon">📊</span>
                <h2>Complexity Analysis</h2>
              </div>
              <div className="header-right">
                <div className="aggregate-preview">
                  <span className="agg-label">Overall</span>
                  <span className="agg-value" style={{ color: getAggregateColor(results.complexity.aggregate) }}>
                    {results.complexity.aggregate.toFixed(2)}
                  </span>
                </div>
                <span className={`chevron ${complexityExpanded ? 'expanded' : ''}`}>▼</span>
              </div>
            </div>

            {complexityExpanded && (
              <div className="card-content">
                {complexityItems.map((item) => (
                  <div key={item.label} className="complexity-row">
                    <div className="complexity-label">
                      <span className="complexity-icon">{item.icon}</span>
                      <span>{item.label}</span>
                    </div>
                    <div className="complexity-bar-container">
                      <div className="complexity-bar-track">
                        <div 
                          className="complexity-bar-fill"
                          style={{ 
                            width: `${Math.min(item.value, 1) * 100}%`,
                            background: `linear-gradient(90deg, ${getScoreColor(item.value)}88, ${getScoreColor(item.value)})`
                          }}
                        />
                      </div>
                      <span className="complexity-value" style={{ color: getScoreColor(item.value) }}>
                        {item.value.toFixed(3)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}

function ResearchTab({ selectedModel, availableModels, onModelChange, llmLoaded, llmLoading, llmProgress, loadLLM }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStatus, setProgressStatus] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const quickPrompts = [
    { text: 'Compare NVDA vs AMD', prompt: 'Compare NVIDIA and AMD stock performance over the last year' },
    { text: 'Analyze AAPL', prompt: 'Analyze Apple stock over the last 6 months' },
    { text: 'Tech comparison', prompt: 'Compare Microsoft, Google, and Meta stock performance over the past year' },
  ];

  const sendMessage = async () => {
    if (!input.trim() || isProcessing) return;
    
    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    
    try {
      const result = await executeResearch(input, (progress) => {
        setProgressStatus(progress.message);
      });
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: result.response,
        chartData: result.chartData,
        toolCalls: result.toolCalls
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}`,
        isError: true
      }]);
    }
    
    setIsProcessing(false);
    setProgressStatus('');
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    clearResearchState();
  };

  const copyMessage = (content) => {
    navigator.clipboard.writeText(content);
  };

  return (
    <div className="research-tab">
      {/* Optional: Model Selector for AI insights */}
      <ModelSelector 
        selectedModel={selectedModel}
        availableModels={availableModels}
        onModelChange={onModelChange}
        llmLoaded={llmLoaded}
        llmLoading={llmLoading}
        llmProgress={llmProgress}
        loadLLM={loadLLM}
      />

      <div className="research-info card">
        <div className="info-header">
          <span>📈</span>
          <h3>Stock Research</h3>
        </div>
        <p>Ask about stock performance, comparisons, and trends. Data from AlphaVantage API.</p>
        {llmLoaded && <p className="ai-note">✨ AI insights enabled</p>}
      </div>

      <div className="chat-container research-container">
        <div className="chat-header">
          <span>📊 Research Chat</span>
          {messages.length > 0 && (
            <button className="clear-btn" onClick={clearChat}>Clear</button>
          )}
        </div>
        
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-chat">
              <span className="empty-icon">📈</span>
              <p>Ask about stocks like "Analyze AAPL" or "Compare NVDA vs AMD"</p>
              <div className="quick-prompts">
                {quickPrompts.map((qp, idx) => (
                  <button 
                    key={idx}
                    className="quick-prompt-btn"
                    onClick={() => setInput(qp.prompt)}
                  >
                    {qp.text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role} ${msg.isError ? 'error' : ''}`}>
                <div className="message-header">
                  <span className="message-role">
                    {msg.role === 'user' ? '👤 You' : '📊 Analyst'}
                  </span>
                  {msg.role === 'assistant' && msg.content && !msg.isError && (
                    <button 
                      className="copy-btn small" 
                      onClick={() => copyMessage(msg.content)}
                      title="Copy"
                    >
                      📋
                    </button>
                  )}
                </div>
                
                {/* Main content */}
                <div className="message-content markdown-content">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
                
                {/* Chart */}
                {msg.chartData && (
                  <div className="chart-container">
                    <StockChart chartData={msg.chartData} />
                  </div>
                )}
                
                {/* Tool calls */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <ToolCallsSection toolCalls={msg.toolCalls} />
                )}
              </div>
            ))
          )}
          
          {isProcessing && (
            <div className="message assistant processing">
              <div className="message-header">
                <span className="message-role">📊 Analyst</span>
              </div>
              <div className="message-content">
                <span className="typing-indicator">●●●</span>
                <span className="progress-status">{progressStatus}</span>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
        
        <div className="chat-input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about stocks... (e.g., Compare NVDA vs AMD)"
            rows={2}
            disabled={isProcessing}
          />
          <button 
            className="send-btn"
            onClick={sendMessage}
            disabled={!input.trim() || isProcessing}
          >
            {isProcessing ? <span className="btn-spinner"></span> : '➤'}
          </button>
        </div>
      </div>
    </div>
  );
}

// Tool Calls Section Component
function ToolCallsSection({ toolCalls }) {
  const [expanded, setExpanded] = useState(false);
  
  const toolIcons = {
    fetchTimeSeries: '📥',
    computeMetrics: '📊',
    compareStocks: '⚖️',
    generateChartData: '📈',
    computeCorrelation: '🔗'
  };
  
  return (
    <div className="tool-calls-section">
      <button className="tool-calls-toggle" onClick={() => setExpanded(!expanded)}>
        <span>🔧 {toolCalls.length} tool{toolCalls.length === 1 ? '' : 's'} called</span>
        <span className={`chevron ${expanded ? 'expanded' : ''}`}>▼</span>
      </button>
      
      {expanded && (
        <div className="tool-calls-list">
          {toolCalls.map((call, idx) => (
            <div key={idx} className="tool-call-item">
              <span className="tool-icon">{toolIcons[call.toolName] || '🔧'}</span>
              <span className="tool-name">{call.toolName}</span>
              <span className="tool-result">{call.result}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Stock Chart Component
function StockChart({ chartData }) {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!canvasRef.current || !chartData) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    
    // Set canvas size
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    
    const width = rect.width;
    const height = rect.height;
    const padding = { top: 20, right: 60, bottom: 40, left: 60 };
    
    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);
    
    // Determine which data to use
    const isPerformance = ['performance', 'normalized', 'comparison'].includes(chartData.chartType?.toLowerCase());
    const dataSource = isPerformance ? chartData.normalizedSeries : chartData.series;
    
    if (!dataSource || Object.keys(dataSource).length === 0) return;
    
    // Get all data points for scaling
    let allValues = [];
    let allDates = [];
    
    Object.values(dataSource).forEach(series => {
      series.forEach(point => {
        allValues.push(isPerformance ? point.value : point.price);
        allDates.push(new Date(point.date).getTime());
      });
    });
    
    const minValue = Math.min(...allValues);
    const maxValue = Math.max(...allValues);
    const minDate = Math.min(...allDates);
    const maxDate = Math.max(...allDates);
    
    const valueRange = maxValue - minValue || 1;
    const dateRange = maxDate - minDate || 1;
    
    // Scale functions
    const scaleX = (date) => padding.left + ((new Date(date).getTime() - minDate) / dateRange) * (width - padding.left - padding.right);
    const scaleY = (value) => height - padding.bottom - ((value - minValue) / valueRange) * (height - padding.top - padding.bottom);
    
    // Draw grid
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    
    // Horizontal grid lines
    const yTicks = 5;
    for (let i = 0; i <= yTicks; i++) {
      const y = padding.top + (i / yTicks) * (height - padding.top - padding.bottom);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      
      // Y-axis labels
      const value = maxValue - (i / yTicks) * valueRange;
      ctx.fillStyle = '#888';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'right';
      if (isPerformance) {
        ctx.fillText(`${value >= 0 ? '+' : ''}${value.toFixed(0)}%`, padding.left - 8, y + 4);
      } else {
        ctx.fillText(`$${value.toFixed(0)}`, padding.left - 8, y + 4);
      }
    }
    
    // Zero line for performance charts
    if (isPerformance && minValue < 0 && maxValue > 0) {
      const zeroY = scaleY(0);
      ctx.strokeStyle = '#666';
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(padding.left, zeroY);
      ctx.lineTo(width - padding.right, zeroY);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    
    // Colors for each ticker
    const colors = ['#3b82f6', '#f97316', '#22c55e', '#a855f7', '#ef4444'];
    
    // Draw lines
    const tickers = Object.keys(dataSource).sort();
    tickers.forEach((ticker, tickerIdx) => {
      const series = dataSource[ticker].sort((a, b) => new Date(a.date) - new Date(b.date));
      
      ctx.strokeStyle = colors[tickerIdx % colors.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      series.forEach((point, i) => {
        const x = scaleX(point.date);
        const y = scaleY(isPerformance ? point.value : point.price);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.stroke();
    });
    
    // Draw legend
    ctx.font = '12px system-ui';
    tickers.forEach((ticker, idx) => {
      const x = padding.left + idx * 80;
      const y = height - 10;
      
      ctx.fillStyle = colors[idx % colors.length];
      ctx.fillRect(x, y - 8, 12, 12);
      
      ctx.fillStyle = '#fff';
      ctx.textAlign = 'left';
      ctx.fillText(ticker, x + 16, y + 2);
    });
    
    // Chart title
    ctx.fillStyle = '#888';
    ctx.font = '11px system-ui';
    ctx.textAlign = 'right';
    ctx.fillText(isPerformance ? 'Performance (% change)' : 'Price ($)', width - padding.right, 14);
    
  }, [chartData]);
  
  return (
    <div className="stock-chart">
      <canvas ref={canvasRef} style={{ width: '100%', height: '200px' }} />
    </div>
  );
}

function ChatTab({ selectedModel, availableModels, onModelChange, llmLoaded, llmLoading, llmProgress, loadLLM }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || isGenerating || !llmLoaded) return;
    
    const userMessage = { role: 'user', content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsGenerating(true);
    
    // Add placeholder for AI response
    const aiMessageIndex = newMessages.length;
    setMessages([...newMessages, { role: 'assistant', content: '', isStreaming: true }]);
    
    try {
      await llm.chat(
        newMessages.map(m => ({ role: m.role, content: m.content })),
        (text) => {
          setMessages(prev => {
            const updated = [...prev];
            updated[aiMessageIndex] = { role: 'assistant', content: text, isStreaming: true };
            return updated;
          });
        }
      );
      
      // Mark as complete
      setMessages(prev => {
        const updated = [...prev];
        updated[aiMessageIndex] = { ...updated[aiMessageIndex], isStreaming: false };
        return updated;
      });
    } catch (err) {
      setMessages(prev => {
        const updated = [...prev];
        updated[aiMessageIndex] = { role: 'assistant', content: `Error: ${err.message}`, isStreaming: false };
        return updated;
      });
    }
    
    setIsGenerating(false);
  };

  const copyMessage = (content) => {
    navigator.clipboard.writeText(content);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="chat-tab">
      {/* Model Selector */}
      <ModelSelector 
        selectedModel={selectedModel}
        availableModels={availableModels}
        onModelChange={onModelChange}
        llmLoaded={llmLoaded}
        llmLoading={llmLoading}
        llmProgress={llmProgress}
        loadLLM={loadLLM}
      />

      {!llmLoaded && !llmLoading && (
        <div className="card info-card">
          <p>👆 Select and load a model above to start chatting</p>
        </div>
      )}

      {llmLoaded && (
        <div className="chat-container">
          <div className="chat-header">
            <span>💬 Chat</span>
            {messages.length > 0 && (
              <button className="clear-btn" onClick={clearChat}>Clear</button>
            )}
          </div>
          
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-chat">
                <span className="empty-icon">💬</span>
                <p>Start a conversation with the AI</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div key={idx} className={`message ${msg.role}`}>
                  <div className="message-header">
                    <span className="message-role">
                      {msg.role === 'user' ? '👤 You' : '🤖 AI'}
                    </span>
                    {msg.role === 'assistant' && msg.content && (
                      <button 
                        className="copy-btn small" 
                        onClick={() => copyMessage(msg.content)}
                        title="Copy"
                      >
                        📋
                      </button>
                    )}
                  </div>
                  <div className="message-content">
                    {msg.content || (msg.isStreaming && <span className="typing-indicator">●●●</span>)}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <div className="chat-input-container">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type a message..."
              rows={2}
              disabled={isGenerating}
            />
            <button 
              className="send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || isGenerating}
            >
              {isGenerating ? <span className="btn-spinner"></span> : '➤'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
