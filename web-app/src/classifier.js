/**
 * Prompt Classifier - On-Device AI Classification
 * 
 * This module provides on-device prompt classification using ONNX Runtime Web.
 * It runs the NVIDIA Prompt Task and Complexity Classifier entirely in the browser,
 * with no server-side processing required.
 * 
 * Model: nvidia/prompt-task-and-complexity-classifier
 * - Architecture: DeBERTa-v3-base with custom classification heads
 * - Converted to ONNX format for browser deployment
 * - ~700MB model size (cached after first load)
 * 
 * Classification Outputs:
 * 1. Task Type: Classifies the prompt into one of 12 categories
 * 2. Complexity Scores: 6 dimensions of prompt complexity
 * 3. Aggregate Score: Weighted combination of complexity dimensions
 * 
 * Technical Stack:
 * - ONNX Runtime Web (WebAssembly backend)
 * - Hugging Face Transformers.js (tokenization)
 * - Cache API (model persistence)
 */

// ============================================
// Model Configuration Constants
// ============================================

/**
 * Task type classification labels
 * Maps model output indices to human-readable task categories
 */
const TASK_TYPE_MAP = {
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
};

/**
 * Weights for computing complexity scores from softmax probabilities
 * These are learned during model training and represent the "score" for each class
 */
const WEIGHTS_MAP = {
  creativity_scope: [2, 1, 0],        // High, Medium, Low creativity
  reasoning: [0, 1],                   // No reasoning, Requires reasoning
  contextual_knowledge: [0, 1],        // No context needed, Context required
  number_of_few_shots: [0, 1, 2, 3, 4, 5],  // Number of examples needed
  domain_knowledge: [3, 1, 2, 0],      // Expert, Intermediate, Basic, None
  no_label_reason: [0],                // Internal use
  constraint_ct: [1, 0]                // Has constraints, No constraints
};

/**
 * Divisors for normalizing weighted scores to 0-1 range
 */
const DIVISOR_MAP = {
  creativity_scope: 2,
  reasoning: 1,
  contextual_knowledge: 1,
  number_of_few_shots: 1,
  domain_knowledge: 3,
  no_label_reason: 1,
  constraint_ct: 1
};

// Cache configuration for model persistence
const MODEL_CACHE_NAME = 'prompt-analyzer-models-v1';
const MODEL_URL = '/models/model.onnx';
const MODEL_SIZE_MB = 702;

/**
 * PromptClassifier - Main classifier class
 * 
 * Handles model loading, tokenization, and inference.
 * Uses singleton pattern via exported instance.
 */
class PromptClassifier {
  constructor() {
    this.session = null;      // ONNX inference session
    this.tokenizer = null;    // HuggingFace tokenizer
    this.isLoaded = false;    // Ready state flag
  }

  /**
   * Fetches a resource with progress tracking and caching
   * 
   * Uses the Cache API to persist the model between page loads,
   * avoiding re-download of the large ONNX model file.
   * 
   * @param {string} url - URL to fetch
   * @param {Function} onProgress - Progress callback ({ loaded, total, cached })
   * @returns {Promise<ArrayBuffer>} The fetched data
   */
  async fetchWithProgress(url, onProgress) {
    // Check browser cache first
    const cache = await caches.open(MODEL_CACHE_NAME);
    const cachedResponse = await cache.match(url);
    
    if (cachedResponse) {
      onProgress?.({ loaded: MODEL_SIZE_MB, total: MODEL_SIZE_MB, cached: true });
      return await cachedResponse.arrayBuffer();
    }
    
    // Fetch with streaming progress
    const response = await fetch(url);
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : MODEL_SIZE_MB * 1024 * 1024;
    
    // Read response body in chunks for progress tracking
    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      chunks.push(value);
      loaded += value.length;
      
      // Report progress in MB
      onProgress?.({
        loaded: loaded / (1024 * 1024),
        total: total / (1024 * 1024),
        cached: false
      });
    }
    
    // Combine chunks into single ArrayBuffer
    const allChunks = new Uint8Array(loaded);
    let position = 0;
    for (const chunk of chunks) {
      allChunks.set(chunk, position);
      position += chunk.length;
    }
    
    // Cache for subsequent page loads
    const responseToCache = new Response(allChunks, {
      headers: { 'Content-Type': 'application/octet-stream' }
    });
    await cache.put(url, responseToCache);
    
    return allChunks.buffer;
  }

  /**
   * Loads the classifier model and tokenizer
   * 
   * Loading sequence:
   * 1. Load tokenizer from Hugging Face (uses their CDN)
   * 2. Fetch ONNX model (from local /models/ or cache)
   * 3. Initialize ONNX Runtime inference session
   * 
   * @param {Function} onProgress - Progress callback for model download
   * @param {Function} onStatusChange - Status message callback
   */
  async load(onProgress, onStatusChange) {
    try {
      // Step 1: Load the DeBERTa tokenizer
      // This handles text → token IDs conversion
      onStatusChange?.('Loading tokenizer...');
      
      const { AutoTokenizer } = await import('@huggingface/transformers');
      
      this.tokenizer = await AutoTokenizer.from_pretrained(
        'nvidia/prompt-task-and-complexity-classifier'
      );
      
      // Step 2: Load the ONNX model
      // Uses global 'ort' from CDN script in index.html
      onStatusChange?.('Loading ONNX model...');
      
      const ort = window.ort;
      if (!ort) {
        throw new Error('ONNX Runtime not loaded. Check index.html script tag.');
      }
      
      // Fetch with progress tracking and caching
      const modelBuffer = await this.fetchWithProgress(MODEL_URL, onProgress);
      
      // Step 3: Create inference session
      onStatusChange?.('Initializing model...');
      
      this.session = await ort.InferenceSession.create(modelBuffer, {
        executionProviders: ['wasm']  // WebAssembly backend for broad compatibility
      });
      
      this.ort = ort;
      this.isLoaded = true;
      onStatusChange?.('Ready!');
      
    } catch (error) {
      console.error('Failed to load model:', error);
      throw error;
    }
  }

  /**
   * Computes softmax probabilities from logits
   * 
   * Softmax converts raw model outputs (logits) to probabilities
   * that sum to 1. Uses the numerically stable version with max subtraction.
   * 
   * Formula: softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
   * 
   * @param {Float32Array} logits - Raw model output values
   * @returns {number[]} Probability distribution
   */
  softmax(logits) {
    const arr = Array.from(logits);
    const maxVal = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sumExps);
  }

  /**
   * Computes a complexity score from model logits
   * 
   * Process:
   * 1. Convert logits to probabilities via softmax
   * 2. Compute weighted sum using pre-defined weights
   * 3. Normalize by divisor to get 0-1 score
   * 
   * @param {Float32Array} logits - Raw model output for this head
   * @param {string} target - Score type (e.g., 'creativity_scope')
   * @returns {number} Normalized score between 0 and 1
   */
  computeScore(logits, target) {
    const probs = this.softmax(logits);
    const weights = WEIGHTS_MAP[target];
    const divisor = DIVISOR_MAP[target];
    
    if (!weights || !divisor) return 0;
    
    // Weighted sum of probabilities
    let weightedSum = 0;
    for (let i = 0; i < probs.length && i < weights.length; i++) {
      weightedSum += probs[i] * weights[i];
    }
    
    let score = weightedSum / divisor;
    
    // Special case: few-shots below threshold treated as 0
    if (target === 'number_of_few_shots' && score < 0.05) {
      score = 0;
    }
    
    return score;
  }

  /**
   * Computes task type classification from logits
   * 
   * Returns the top predicted task type, secondary prediction (if confident),
   * and full probability distribution for visualization.
   * 
   * @param {Float32Array} logits - Task type head output
   * @returns {Object} Task type results with probabilities
   */
  computeTaskType(logits) {
    const probs = this.softmax(logits);
    
    // Build full probability map for charts
    const allProbabilities = {};
    probs.forEach((prob, idx) => {
      allProbabilities[TASK_TYPE_MAP[idx]] = prob;
    });
    
    // Sort to find top predictions
    const indexed = probs.map((prob, idx) => ({ prob, idx }))
      .sort((a, b) => b.prob - a.prob);
    
    const top1 = indexed[0];
    const top2 = indexed[1];
    
    return {
      taskType1: TASK_TYPE_MAP[top1.idx],
      taskType2: top2.prob >= 0.1 ? TASK_TYPE_MAP[top2.idx] : null,  // Only show if >10%
      probability: top1.prob,
      allProbabilities
    };
  }

  /**
   * Runs inference on a text prompt
   * 
   * Pipeline:
   * 1. Tokenize input text (pad/truncate to 128 tokens)
   * 2. Create ONNX tensors for input_ids and attention_mask
   * 3. Run model inference
   * 4. Process outputs from all classification heads
   * 5. Compute aggregate complexity score
   * 
   * @param {string} text - The prompt to classify
   * @returns {Promise<Object>} Classification results
   */
  async predict(text) {
    if (!this.isLoaded) {
      throw new Error('Model not loaded');
    }

    // Tokenize input - converts text to token IDs
    // DeBERTa uses SentencePiece tokenization
    const encoded = await this.tokenizer(text, {
      padding: 'max_length',
      truncation: true,
      max_length: 128
    });

    // Create ONNX tensors
    // Model expects int64 tensors with shape [batch_size, sequence_length]
    const inputIds = new this.ort.Tensor('int64', 
      BigInt64Array.from(encoded.input_ids.data.map(x => BigInt(x))),
      [1, 128]
    );
    
    const attentionMask = new this.ort.Tensor('int64',
      BigInt64Array.from(encoded.attention_mask.data.map(x => BigInt(x))),
      [1, 128]
    );

    // Run inference - model has multiple output heads
    const outputs = await this.session.run({
      input_ids: inputIds,
      attention_mask: attentionMask
    });

    // Process each classification head
    const taskType = this.computeTaskType(outputs.logits_task_type.data);
    const creativity = this.computeScore(outputs.logits_creativity_scope.data, 'creativity_scope');
    const reasoning = this.computeScore(outputs.logits_reasoning.data, 'reasoning');
    const contextual = this.computeScore(outputs.logits_contextual_knowledge.data, 'contextual_knowledge');
    const fewShots = this.computeScore(outputs.logits_few_shots.data, 'number_of_few_shots');
    const domain = this.computeScore(outputs.logits_domain_knowledge.data, 'domain_knowledge');
    const constraint = this.computeScore(outputs.logits_constraint_ct.data, 'constraint_ct');

    // Compute weighted aggregate score
    // Weights reflect relative importance of each dimension
    const aggregate = 
      0.35 * creativity +   // Creativity is most important
      0.25 * reasoning +    // Reasoning second
      0.15 * constraint +   // Constraints matter
      0.15 * domain +       // Domain knowledge
      0.05 * contextual +   // Context requirements
      0.05 * fewShots;      // Few-shot needs

    return {
      taskType,
      complexity: {
        creativity,
        reasoning,
        contextual,
        domain,
        constraint,
        fewShots,
        aggregate
      }
    };
  }
}

// Export singleton instance
export const classifier = new PromptClassifier();
