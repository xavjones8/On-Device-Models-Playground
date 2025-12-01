//
//  OnDeviceTestApp.swift
//  OnDeviceTest
//
//  Created by Xavier Jones on 11/29/25.
//

/*
 On-Device AI Demo App
 
 This app demonstrates on-device AI capabilities using:
 - CoreML: NVIDIA Prompt Classifier for task/complexity analysis
 - Apple Foundation Models: On-device LLM for chat and responses
 - AlphaVantage API: Stock data for tool-calling demonstration
 
 Project Structure:
 ├── App/           - App entry point
 ├── Views/         - SwiftUI views (Analyzer, Chat, Research, Debug)
 ├── Models/        - Data models and enums
 ├── Services/
 │   ├── PromptClassifier/  - CoreML model + tokenizer
 │   └── StockAnalysis/     - AlphaVantage API + tools
 └── Resources/     - ML models, tokenizer files
 */

import SwiftUI

@main
struct OnDeviceTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
