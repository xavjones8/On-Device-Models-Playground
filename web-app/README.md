# Web App - On-Device AI Demo

React web application demonstrating browser-based AI with ONNX classification and WebLLM chat.

## Requirements

- Node.js 18+
- Modern browser with WebGPU support:
  - Chrome 113+ (recommended)
  - Edge 113+
  - Firefox Nightly (experimental)

## Quick Start

```bash
# Install dependencies
npm install

# Create environment file
cp .env.example .env
# Edit .env with your API keys

# Start development server
npm run dev

# Build for production
npm run build
```

## Project Structure

```
web-app/
├── src/
│   ├── App.jsx           # Main React component (tabs, UI)
│   ├── App.css           # Styling (dark theme)
│   ├── classifier.js     # ONNX model wrapper
│   ├── llm.js            # WebLLM wrapper
│   ├── stockService.js   # AlphaVantage API client
│   ├── stockResearch.js  # Stock analysis agent
│   ├── main.jsx          # React entry point
│   └── index.css         # Global styles
├── public/
│   └── models/           # ONNX model files
│       ├── model.onnx
│       ├── tokenizer.json
│       └── classifier_metadata.json
├── .env.example          # Environment template
└── vite.config.js        # Vite configuration
```

## Features

### 1. Analyzer Tab
- Prompt classification using ONNX model
- Task type prediction (12 categories)
- Complexity scores (6 dimensions)
- AI response generation via WebLLM

### 2. Chat Tab
- Multi-turn conversation
- Model selection from available WebLLM models
- Streaming responses
- Markdown rendering

### 3. Research Tab
- Natural language stock queries
- Real-time data from AlphaVantage
- Financial metrics computation
- Markdown-formatted reports

## Configuration

### Environment Variables

Create `.env` file:
```env
VITE_ALPHAVANTAGE_API_KEY=your_api_key_here
```

Get a free key at: https://www.alphavantage.co/support/#api-key

### WebLLM Models

Available models are filtered for browser compatibility. The app auto-loads the first available model on startup.

Supported models include:
- Llama-3.2-1B-Instruct
- Llama-3.2-3B-Instruct  
- Mistral-7B-Instruct
- Phi-3.5-mini-instruct
- Qwen2.5-1.5B-Instruct

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   React UI                       │
│              (App.jsx + App.css)                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │  Analyzer   │  │    Chat     │  │ Research │ │
│  │    Tab      │  │    Tab      │  │   Tab    │ │
│  └──────┬──────┘  └──────┬──────┘  └────┬─────┘ │
│         │                │               │       │
└─────────┼────────────────┼───────────────┼───────┘
          │                │               │
          ▼                ▼               ▼
   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
   │classifier.js│  │   llm.js    │  │stockResearch │
   │   (ONNX)    │  │  (WebLLM)   │  │     .js      │
   └──────┬──────┘  └──────┬──────┘  └──────┬───────┘
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
   │ONNX Runtime │  │   WebGPU    │  │stockService  │
   │    Web      │  │   Engine    │  │     .js      │
   └─────────────┘  └─────────────┘  └──────┬───────┘
                                            │
                                            ▼
                                     ┌──────────────┐
                                     │ AlphaVantage │
                                     │     API      │
                                     └──────────────┘
```

## Key Files

### `classifier.js`
- Loads ONNX model with caching (Cache API)
- Tokenizes input using @huggingface/transformers
- Runs inference via ONNX Runtime Web
- Post-processes logits to scores

### `llm.js`
- Initializes WebLLM engine
- Manages model loading with progress
- Handles streaming generation
- Provides available model list

### `stockService.js`
- AlphaVantage API client
- Financial metric calculations (CAGR, volatility)
- Correlation computation
- Data caching

### `stockResearch.js`
- Natural language query parsing
- Tool orchestration
- Report generation
- AI insight integration

## Development

```bash
# Start dev server with hot reload
npm run dev

# Lint code
npm run lint

# Build for production
npm run build

# Preview production build
npm run preview
```

## Troubleshooting

### "WebGPU not supported"
- Use Chrome 113+ or Edge 113+
- Enable WebGPU in browser flags if needed
- Check GPU driver is up to date

### Model loading slow
- First load downloads ~1-4GB depending on model
- Subsequent loads use cached version
- Progress bar shows download status

### ONNX errors
- Ensure ONNX Runtime loaded from CDN
- Check browser console for specific errors
- Verify model files in public/models/

### Stock data not loading
- Check API key in .env file
- Free tier: 5 calls/min, 500/day
- Try different ticker symbols
