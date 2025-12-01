# On-Device AI Demo

A demonstration of on-device AI capabilities across iOS and Web platforms, featuring prompt classification, LLM chat, and tool-calling with stock analysis.

## Project Structure

```
on-device/
├── OnDeviceTest/          # iOS/macOS app (Swift/SwiftUI)
├── web-app/               # Web application (React/Vite)
├── model_converters/      # Python scripts for model conversion
├── onnx_model/            # Converted ONNX model files
└── PromptGrader.mlpackage # Converted CoreML model
```

## Features

### 1. Prompt Classification
- **Model**: NVIDIA prompt-task-and-complexity-classifier
- **Task Types**: 12 categories (Code Generation, Text Generation, QA, etc.)
- **Complexity Scores**: 6 dimensions (creativity, reasoning, constraints, etc.)

### 2. On-Device LLM Chat
- **iOS**: Apple Foundation Models (Apple Intelligence)
- **Web**: WebLLM (various open-source models via WebGPU)

### 3. Tool Calling (Stock Research)
- **API**: AlphaVantage for stock market data
- **Tools**: Fetch data, compute metrics, compare stocks, generate charts
- **Visualization**: Swift Charts (iOS) / Canvas (Web)

## Quick Start

### iOS App
1. Open `OnDeviceTest/OnDeviceTest.xcodeproj` in Xcode
2. Build and run on device (requires Apple Intelligence)
3. See `OnDeviceTest/README.md` for details

### Web App
```bash
cd web-app
npm install
npm run dev
```
See `web-app/README.md` for details

### Model Conversion
```bash
cd model_converters
pip install -r requirements.txt
python coreml.py      # For iOS
python onnx_convert.py # For Web
```
See `model_converters/README.md` for details

## Requirements

### iOS
- Xcode 16+
- iOS 18.1+ / macOS 15.1+
- Device with Apple Intelligence support

### Web
- Node.js 18+
- Modern browser with WebGPU support (Chrome 113+)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface                          │
├─────────────────────────────────────────────────────────────┤
│  iOS: SwiftUI          │  Web: React + Vite                 │
├─────────────────────────────────────────────────────────────┤
│                   Classification Layer                       │
├─────────────────────────────────────────────────────────────┤
│  iOS: CoreML           │  Web: ONNX Runtime                 │
│  PromptGrader.mlpackage│  model.onnx                        │
├─────────────────────────────────────────────────────────────┤
│                      LLM Layer                              │
├─────────────────────────────────────────────────────────────┤
│  iOS: Apple Foundation │  Web: WebLLM                       │
│  Models                │  (Llama, Mistral, etc.)            │
├─────────────────────────────────────────────────────────────┤
│                    Tool Layer                               │
├─────────────────────────────────────────────────────────────┤
│  AlphaVantage API for stock data                            │
│  Compute metrics, correlations, charts                      │
└─────────────────────────────────────────────────────────────┘
```

## License

This project uses the NVIDIA prompt classifier model, subject to NVIDIA's license terms.
AlphaVantage API usage subject to their terms of service.


