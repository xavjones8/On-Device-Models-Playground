# Views

SwiftUI views for the app's user interface.

## Files

### `ContentView.swift`
Main entry view with tab navigation. Also contains `PromptAnalyzerView` for the Analyzer tab.

**Components:**
- `ContentView` - TabView with 4 tabs
- `PromptAnalyzerView` - Prompt input, classification display, AI response
- `TaskClassificationCard` - Collapsible task type display
- `ComplexityScoresCard` - Collapsible complexity scores
- `ScoreRow` - Individual score display with progress bar

### `ChatView.swift`
General-purpose chat interface with Apple Foundation Models.

**Components:**
- `ChatView` - Main chat UI
- `ChatViewModel` - Manages conversation state
- `MessageBubble` - Individual message display
- `AvailabilityBanner` - Shows Apple Intelligence status

**Features:**
- Multi-turn conversation
- Streaming responses
- Markdown rendering (MarkdownUI)
- Copy functionality (button + long-press)

### `StockResearchView.swift`
Stock analysis chat with tool-calling capabilities.

**Components:**
- `StockResearchView` - Main research UI
- `StockResearchViewModel` - Manages session with tools
- `StockUserBubble` / `StockAIBubble` - Message bubbles
- `EmbeddedChartView` - Swift Charts visualization
- `WelcomeCard` / `QuickPromptButton` - Onboarding UI

**Features:**
- Natural language stock queries
- Automatic tool invocation
- Interactive charts
- Tool call visualization (collapsible)

### `APIDebugView.swift`
Debug interface for testing AlphaVantage API directly.

**Components:**
- `APIDebugView` - Main debug UI
- `APIDebugViewModel` - Manages API calls
- `MetricRow` - Formatted metric display

**Features:**
- Direct API testing
- Raw JSON response display
- Metric computation verification
- Chart preview

## Common Patterns

### State Management
- `@StateObject` for view models
- `@Published` for observable properties
- `@State` for local UI state

### Async Operations
- `Task { }` for async work
- Loading states with `isProcessing`
- Error handling with optional `errorMessage`

### Styling
- Consistent card backgrounds (`Color(UIColor.secondarySystemBackground)`)
- Rounded corners (12-18pt)
- SF Symbols for icons


