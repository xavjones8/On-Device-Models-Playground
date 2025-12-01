# Models

Shared data models and enums used across the app.

## Files

### `FoundationModelAvailability.swift`
Enum representing Apple Intelligence availability status.

```swift
enum FoundationModelAvailability {
    case available       // Ready to use
    case notEligible     // Device doesn't support Apple Intelligence
    case notEnabled      // User hasn't enabled Apple Intelligence
    case notReady        // Model is downloading/preparing
    case unknown         // Status couldn't be determined
}
```

**Usage:**
- Check before attempting to use Foundation Models
- Display appropriate error messages to users
- Guide users to enable Apple Intelligence if needed

## Adding New Models

When adding new shared data structures:

1. Create a new `.swift` file in this folder
2. Use `struct` for value types, `class` for reference types
3. Conform to `Codable` if JSON serialization needed
4. Conform to `Identifiable` if used in SwiftUI lists
5. Add documentation comments

## Example

```swift
// NewModel.swift

/// Represents a user's analysis history entry
struct AnalysisHistoryEntry: Identifiable, Codable {
    let id: UUID
    let prompt: String
    let timestamp: Date
    let scores: ComplexityScores
    
    init(prompt: String, scores: ComplexityScores) {
        self.id = UUID()
        self.prompt = prompt
        self.timestamp = Date()
        self.scores = scores
    }
}
```


