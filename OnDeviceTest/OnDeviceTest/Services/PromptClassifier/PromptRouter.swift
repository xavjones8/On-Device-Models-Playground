/*
 PromptRouter.swift
 
 This file contains the core prompt classification and routing logic:
 
 1. PromptClassifier: Wraps the CoreML model (PromptGrader.mlpackage)
    - Tokenizes input using DebertaTokenizer
    - Runs inference on the CoreML model
    - Post-processes logits to compute scores
 
 2. FoundationModelGenerator: Wraps Apple's on-device LLM
    - Generates responses using Apple Intelligence
    - Supports streaming for real-time output
 
 3. PromptRouter: Orchestrates classification and generation
    - Observable object for SwiftUI binding
    - Coordinates classifier and generator
 
 Classification Outputs:
 - Task Type: 12 categories (Code Gen, Text Gen, QA, etc.)
 - Complexity Scores: 6 dimensions (creativity, reasoning, etc.)
 - Aggregate Score: Weighted combination for routing decisions
 
 Post-Processing:
 The CoreML model outputs raw logits. We apply:
 1. Softmax to convert to probabilities
 2. Weighted sum using weightsMap
 3. Normalization using divisorMap
 */

import Foundation
import CoreML
import Combine
import SwiftUI
import FoundationModels

// MARK: - Data Structures

struct TaskTypePrediction {
    let taskType1: String
    let taskType2: String
    let probability: Double
    let allProbabilities: [String: Double]
}

struct ComplexityScores {
    let creativity: Double
    let reasoning: Double
    let constraint: Double
    let domainKnowledge: Double
    let contextualKnowledge: Double
    let fewShots: Double
    
    // Task type prediction
    let taskType: TaskTypePrediction
    
    /// A composite score useful for routing decisions
    var aggregate: Double {
        return 0.35 * creativity
             + 0.25 * reasoning
             + 0.15 * constraint
             + 0.15 * domainKnowledge
             + 0.05 * contextualKnowledge
             + 0.05 * fewShots
    }
}

// MARK: - Prompt Classifier (CoreML Wrapper)

class PromptClassifier {
    let model: PromptGrader
    let tokenizer: DebertaTokenizer
    
    // Configuration from the model's config.json (Weights & Divisors)
    private let weightsMap: [String: [Double]] = [
        "creativity_scope": [2, 1, 0], // 
        "reasoning": [0, 1],
        "contextual_knowledge": [0, 1],
        "number_of_few_shots": [0, 1, 2, 3, 4, 5],
        "domain_knowledge": [3, 1, 2, 0],
        "no_label_reason": [0],
        "constraint_ct": [1, 0]
    ]
    
    private let divisorMap: [String: Double] = [
        "creativity_scope": 2,
        "reasoning": 1,
        "contextual_knowledge": 1,
        "number_of_few_shots": 1,
        "domain_knowledge": 3,
        "no_label_reason": 1,
        "constraint_ct": 1
    ]
    
    // Task type mapping (12 classes) - from config.task_type_map
    private let taskTypeMap: [Int: String] = [
        0: "Brainstorming",
        1: "Chatbot",
        2: "Classification",
        3: "Closed QA",
        4: "Code Generation",
        5: "Extraction",
        6: "Open QA",
        7: "Other",
        8: "Rewrite",
        9: "Summarization",
        10: "Text Generation",
        11: "Unknown"
    ]
    
    init() async throws {
        // 1. Load the CoreML model with optimized settings
        let config = MLModelConfiguration()
        // Use CPU + Neural Engine for best balance of speed and efficiency
        config.computeUnits = .cpuAndNeuralEngine
        
        // Load asynchronously on background thread
        self.model = try await Task.detached(priority: .userInitiated) {
            try PromptGrader(configuration: config)
        }.value
        
        // 2. Load the custom DeBERTa tokenizer
        self.tokenizer = try DebertaTokenizer()
    }
    
    func predict(prompt: String) async throws -> ComplexityScores {
        // 1. Tokenize using our custom DeBERTa tokenizer
        let inputIds = tokenizer.encode(text: prompt)
        
        // 2. Prepare CoreML Inputs
        let sequenceLength = 128
        guard let inputIdsArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32),
              let maskArray = try? MLMultiArray(shape: [1, NSNumber(value: sequenceLength)], dataType: .int32) else {
            throw NSError(domain: "PromptClassifier", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to create MLMultiArrays"])
        }
        
        for i in 0..<sequenceLength {
            let index = [0, NSNumber(value: i)] as [NSNumber]
            if i < inputIds.count {
                inputIdsArray[index] = NSNumber(value: Int32(inputIds[i]))
                maskArray[index] = NSNumber(value: 1)
            } else {
                inputIdsArray[index] = NSNumber(value: 0)
                maskArray[index] = NSNumber(value: 0)
            }
        }
        
        // 3. Run Prediction
        let input = PromptGraderInput(input_ids: inputIdsArray, attention_mask: maskArray)
        let output = try await model.prediction(input: input)
        
        // 4. Process Logits & Calculate Scores
        let creativity = computeScore(logits: output.logits_creativity_scope, target: "creativity_scope")
        let reasoning = computeScore(logits: output.logits_reasoning, target: "reasoning")
        let constraint = computeScore(logits: output.logits_constraint_ct, target: "constraint_ct")
        let domain = computeScore(logits: output.logits_domain_knowledge, target: "domain_knowledge")
        let contextual = computeScore(logits: output.logits_contextual_knowledge, target: "contextual_knowledge")
        let fewShots = computeScore(logits: output.logits_few_shots, target: "number_of_few_shots")
        
        // 5. Process Task Type
        let taskType = computeTaskType(logits: output.logits_task_type)
        
        return ComplexityScores(
            creativity: creativity,
            reasoning: reasoning,
            constraint: constraint,
            domainKnowledge: domain,
            contextualKnowledge: contextual,
            fewShots: fewShots,
            taskType: taskType
        )
    }
    
    /// Helper to compute weighted score from logits
    private func computeScore(logits: MLMultiArray, target: String) -> Double {
        // 1. Softmax
        let count = logits.count
        var exps = [Double]()
        var sumExp = 0.0
        
        for i in 0..<count {
            let val = logits[i].doubleValue
            let e = exp(val)
            exps.append(e)
            sumExp += e
        }
        
        let probs = exps.map { $0 / sumExp }
        
        // 2. Apply Weights
        guard let weights = weightsMap[target], let divisor = divisorMap[target] else {
            return 0.0
        }
        
        var weightedSum = 0.0
        for (i, prob) in probs.enumerated() {
            if i < weights.count {
                weightedSum += prob * weights[i]
            }
        }
        
        // 3. Apply Divisor
        var score = weightedSum / divisor
        
        // Special handling for few shots
        if target == "number_of_few_shots" {
            if score < 0.05 { score = 0.0 }
        }
        
        return score
    }
    
    /// Compute task type prediction from logits
    private func computeTaskType(logits: MLMultiArray) -> TaskTypePrediction {
        // 1. Softmax
        let count = logits.count
        var exps = [Double]()
        var sumExp = 0.0
        
        for i in 0..<count {
            let val = logits[i].doubleValue
            let e = exp(val)
            exps.append(e)
            sumExp += e
        }
        
        let probs = exps.map { $0 / sumExp }
        
        // 2. Build all probabilities dictionary
        var allProbs: [String: Double] = [:]
        for (idx, prob) in probs.enumerated() {
            if let taskName = taskTypeMap[idx] {
                allProbs[taskName] = prob
            }
        }
        
        // 3. Get top 2 indices
        let indexed = probs.enumerated().sorted { $0.element > $1.element }
        let top1Idx = indexed[0].offset
        let top2Idx = indexed[1].offset
        let top1Prob = indexed[0].element
        let top2Prob = indexed[1].element
        
        let taskType1 = taskTypeMap[top1Idx] ?? "Unknown"
        // Only show second task type if probability >= 0.1
        let taskType2 = top2Prob >= 0.1 ? (taskTypeMap[top2Idx] ?? "NA") : "NA"
        
        return TaskTypePrediction(
            taskType1: taskType1,
            taskType2: taskType2,
            probability: top1Prob,
            allProbabilities: allProbs
        )
    }
}

// MARK: - Foundation Model Response Generator

@MainActor
class FoundationModelGenerator: ObservableObject {
    @Published var isGenerating: Bool = false
    @Published var generatedResponse: String = ""
    @Published var availability: FoundationModelAvailability = .unknown
    @Published var errorMessage: String?
    
    private var session: LanguageModelSession?
    
    init() {
        checkAvailability()
        setupSession()
    }
    
    private func setupSession() {
        guard availability == .available else { return }
        
        let instructions = """
            You are a helpful assistant. Respond concisely and directly to the user's request.
            Keep responses focused and avoid unnecessary elaboration unless the task requires it.
            """
        
        session = LanguageModelSession(instructions: instructions)
    }
    
    func checkAvailability() {
        let model = SystemLanguageModel.default
        switch model.availability {
        case .available:
            availability = .available
        case .unavailable(.deviceNotEligible):
            availability = .notEligible
        case .unavailable(.appleIntelligenceNotEnabled):
            availability = .notEnabled
        case .unavailable(.modelNotReady):
            availability = .notReady
        default:
            availability = .unknown
        }
    }
    
    func generateResponse(to prompt: String) async {
        guard availability == .available else {
            errorMessage = "Apple Intelligence is not available on this device"
            return
        }
        
        guard let session = session else {
            errorMessage = "AI session failed to initialize"
            return
        }
        
        isGenerating = true
        generatedResponse = ""
        errorMessage = nil
        
        do {
            let options = GenerationOptions(temperature: 0.7)
            
            // Use streaming to show progressive results
            let stream = session.streamResponse(
                to: prompt,
                options: options
            )
            
            for try await partialResponse in stream {
                generatedResponse = partialResponse.content
            }
        } catch let error as LanguageModelSession.GenerationError {
            errorMessage = "Generation failed: \(error.localizedDescription)"
        } catch {
            errorMessage = "Failed to generate response: \(error.localizedDescription)"
        }
        
        isGenerating = false
    }
}

// MARK: - Prompt Router

enum ModelDestination {
    case onDevice
    case cloud
}

@MainActor
class PromptRouter: ObservableObject {
    @Published var intelligenceThreshold: Double = 0.5
    @Published var lastScores: ComplexityScores?
    @Published var lastDestination: ModelDestination?
    @Published var isGrading: Bool = false
    @Published var isModelLoading: Bool = true
    @Published var errorMessage: String?
    
    // Foundation Model properties (forwarded for proper observation)
    @Published var foundationResponse: String = ""
    @Published var foundationIsGenerating: Bool = false
    @Published var foundationAvailability: FoundationModelAvailability = .unknown
    @Published var foundationErrorMessage: String?
    
    private var grader: PromptClassifier?
    private let foundationModel = FoundationModelGenerator()
    private var cancellables = Set<AnyCancellable>()
    
    init() {
        // Forward foundation model updates to published properties
        foundationModel.$generatedResponse
            .sink { [weak self] value in
                self?.foundationResponse = value
            }
            .store(in: &cancellables)
        
        foundationModel.$isGenerating
            .sink { [weak self] value in
                self?.foundationIsGenerating = value
            }
            .store(in: &cancellables)
        
        foundationModel.$availability
            .sink { [weak self] value in
                self?.foundationAvailability = value
            }
            .store(in: &cancellables)
        
        foundationModel.$errorMessage
            .sink { [weak self] value in
                self?.foundationErrorMessage = value
            }
            .store(in: &cancellables)
        
        Task {
            do {
                self.grader = try await PromptClassifier()
                self.isModelLoading = false
            } catch {
                print("Failed to init grader: \(error)")
                self.errorMessage = error.localizedDescription
                self.isModelLoading = false
            }
        }
    }
    
    func route(prompt: String) async -> ModelDestination {
        guard let grader = grader else { return .cloud }
        isGrading = true
        defer { isGrading = false }
        
        do {
            let scores = try await grader.predict(prompt: prompt)
            lastScores = scores
            let complexity = scores.aggregate
            let dest: ModelDestination = complexity > intelligenceThreshold ? .cloud : .onDevice
            lastDestination = dest
            return dest
        } catch {
            errorMessage = error.localizedDescription
            return .cloud
        }
    }
    
    func analyzeAndRespond(prompt: String) async {
        // First, classify the prompt
        _ = await route(prompt: prompt)
        
        // Then generate a response using Foundation Models
        await foundationModel.generateResponse(to: prompt)
    }
}
