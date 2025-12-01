# OnDeviceTest - iOS App

SwiftUI app demonstrating on-device AI with CoreML classification, Apple Foundation Models, and tool-calling.

## Requirements

- Xcode 16+
- iOS 18.1+ / macOS 15.1+
- Device with Apple Intelligence support:
  - iPhone 15 Pro or later
  - iPad with M1 chip or later
  - Mac with Apple Silicon

## Project Structure

```
OnDeviceTest/
├── App/
│   └── OnDeviceTestApp.swift       # App entry point
├── Views/
│   ├── ContentView.swift           # Tab navigation + Analyzer view
│   ├── ChatView.swift              # General AI chat
│   ├── StockResearchView.swift     # Stock analysis with tools
│   └── APIDebugView.swift          # Debug interface
├── Models/
│   └── FoundationModelAvailability.swift  # Shared availability enum
├── Services/
│   ├── PromptClassifier/
│   │   ├── PromptRouter.swift      # CoreML wrapper + orchestration
│   │   └── DebertaTokenizer.swift  # Custom SentencePiece tokenizer
│   └── StockAnalysis/
│       ├── AlphaVantageService.swift   # Stock API client
│       └── StockAnalysisTools.swift    # Tool definitions
└── Resources/
    ├── PromptGrader.mlpackage      # CoreML classification model
    ├── tokenizer.json              # Tokenizer vocabulary
    ├── tokenizer_config.json       # Tokenizer settings
    └── vocab.json                  # Vocabulary file
```

## Features

### 1. Analyzer Tab
- Enter a prompt to analyze
- View task type classification (12 categories)
- View complexity scores (6 dimensions)
- Get AI-generated response

### 2. Chat Tab
- Multi-turn conversation
- Streaming responses
- Markdown rendering
- Copy functionality

### 3. Research Tab
- Natural language stock queries
- Automatic tool invocation
- Interactive Swift Charts
- Tool call visualization

### 4. Debug Tab
- Direct API testing
- Raw response inspection
- Metric verification

## Dependencies

Managed via Swift Package Manager:
- `swift-transformers` - Tokenizer utilities
- `MarkdownUI` - Markdown rendering

## Setup

1. Open `OnDeviceTest.xcodeproj` in Xcode
2. Select your target device
3. Build and run (⌘R)

Note: Apple Intelligence must be enabled in Settings > Apple Intelligence & Siri

## API Keys

For stock research features, set your AlphaVantage API key:
- Environment variable: `ALPHAVANTAGE_API_KEY`
- Or hardcoded in `AlphaVantageService.swift` (not recommended for production)

Get a free key at: https://www.alphavantage.co/support/#api-key

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  SwiftUI Views                   │
├─────────────────────────────────────────────────┤
│  ContentView │ ChatView │ StockResearchView     │
└──────┬───────┴────┬─────┴──────────┬────────────┘
       │            │                │
       ▼            ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│PromptRouter  │ │ ChatViewModel│ │StockResearch │
│              │ │              │ │  ViewModel   │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│PromptClassif │ │LanguageModel │ │LanguageModel │
│(CoreML)      │ │   Session    │ │Session+Tools │
└──────────────┘ └──────────────┘ └──────────────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │AlphaVantage  │
                                  │   Service    │
                                  └──────────────┘
```

## Troubleshooting

### "Apple Intelligence not available"
- Ensure device meets requirements
- Enable in Settings > Apple Intelligence & Siri
- Wait for model download to complete

### "DeBERTaV2Tokenizer not supported"
- This is handled by custom `DebertaTokenizer`
- Ensure `vocab.json` is in Resources folder

### Stock data not loading
- Check API key is valid
- Free tier: 5 calls/min, 500/day
- Use Debug tab to test API directly


