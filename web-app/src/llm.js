/**
 * WebLLM Service - On-Device Large Language Model
 * 
 * This module provides on-device LLM inference using WebLLM (MLC AI).
 * Models run entirely in the browser using WebGPU acceleration,
 * with no server-side processing required.
 * 
 * Technology Stack:
 * - WebLLM (MLC AI): Compiles LLMs to run in browser via WebGPU
 * - WebGPU: GPU acceleration for fast inference
 * - Model caching: Models are cached in IndexedDB after first download
 * 
 * Supported Models:
 * - Llama family (various sizes)
 * - Mistral family
 * - Gemma (Google)
 * - Phi (Microsoft)
 * - Qwen (Alibaba)
 * - SmolLM (Hugging Face)
 * 
 * Note: Requires WebGPU support (Chrome 113+, Edge 113+)
 * Safari and Firefox have limited WebGPU support.
 */

import * as webllm from '@mlc-ai/web-llm';

/**
 * Gets a curated list of available models from WebLLM
 * 
 * Filters the full model list to include only:
 * - Popular model families (Llama, Mistral, Gemma, Phi, Qwen, SmolLM)
 * - Quantized versions (Q4 for smaller size, FP16 for quality)
 * 
 * Models are sorted by estimated size (smallest first) for
 * better user experience on devices with limited resources.
 * 
 * @returns {Array<Object>} Array of model objects with id, name, size, family
 */
export const getAvailableModels = () => {
  // WebLLM provides a prebuilt config with all available models
  const allModels = webllm.prebuiltAppConfig.model_list;
  
  // Filter to practical browser-compatible models
  const recommended = allModels
    .filter(m => {
      const id = m.model_id.toLowerCase();
      
      // Include popular model families
      const isPopularFamily = (
        id.includes('llama') ||
        id.includes('mistral') ||
        id.includes('gemma') ||
        id.includes('phi') ||
        id.includes('qwen') ||
        id.includes('smollm')
      );
      
      // Prefer quantized versions for reasonable download sizes
      const isQuantized = (
        id.includes('q4f16') ||   // 4-bit quantization, FP16 activations
        id.includes('q4f32') ||   // 4-bit quantization, FP32 activations
        id.includes('q0f16') ||   // No quantization, FP16
        id.includes('q0f32')      // No quantization, FP32
      );
      
      return isPopularFamily && isQuantized;
    })
    .map(m => ({
      id: m.model_id,
      name: formatModelName(m.model_id),
      size: estimateSize(m.model_id),
      family: getModelFamily(m.model_id)
    }))
    .sort((a, b) => {
      // Sort by size (smaller first for faster loading)
      const sizeOrder = { 'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4 };
      return (sizeOrder[a.size] || 5) - (sizeOrder[b.size] || 5);
    });

  return recommended;
};

/**
 * Formats a model ID into a human-readable name
 * 
 * Transforms: "Llama-3.2-1B-Instruct-q4f16_1-MLC"
 * Into: "Llama 3.2 1B (Q4)"
 * 
 * @param {string} modelId - Raw model identifier
 * @returns {string} Formatted display name
 */
function formatModelName(modelId) {
  return modelId
    .replace(/-MLC$/, '')           // Remove MLC suffix
    .replace(/_/g, ' ')             // Underscores to spaces
    .replace(/-/g, ' ')             // Dashes to spaces
    .replace(/q4f16/gi, '(Q4)')     // Quantization labels
    .replace(/q4f32/gi, '(Q4-32)')
    .replace(/q0f16/gi, '(FP16)')
    .replace(/q0f32/gi, '(FP32)')
    .replace(/(\d)B/g, '$1B')       // Ensure "B" attached to number
    .replace(/Instruct/gi, '')      // Remove "Instruct" for brevity
    .trim();
}

/**
 * Determines the model family from the model ID
 * 
 * @param {string} modelId - Model identifier
 * @returns {string} Family name (Llama, Mistral, etc.)
 */
function getModelFamily(modelId) {
  const id = modelId.toLowerCase();
  if (id.includes('llama')) return 'Llama';
  if (id.includes('mistral')) return 'Mistral';
  if (id.includes('gemma')) return 'Gemma';
  if (id.includes('phi')) return 'Phi';
  if (id.includes('qwen')) return 'Qwen';
  if (id.includes('smollm')) return 'SmolLM';
  return 'Other';
}

/**
 * Estimates model download size category based on parameter count
 * 
 * Size categories:
 * - XS: < 500MB (0.1B-0.5B params)
 * - S: ~1GB (1B-2B params)
 * - M: ~2GB (3B-4B params)
 * - L: ~4GB (7B-8B params)
 * - XL: > 4GB (larger models)
 * 
 * @param {string} modelId - Model identifier
 * @returns {string} Size category
 */
function estimateSize(modelId) {
  const id = modelId.toLowerCase();
  
  // Match parameter count patterns in model name
  if (id.includes('0.5b') || id.includes('0.1b') || id.includes('135m') || id.includes('360m')) {
    return 'XS';
  }
  if (id.includes('1b') || id.includes('1.5b') || id.includes('2b')) {
    return 'S';
  }
  if (id.includes('3b') || id.includes('4b')) {
    return 'M';
  }
  if (id.includes('7b') || id.includes('8b')) {
    return 'L';
  }
  return 'XL';
}

/**
 * LLMService - WebLLM wrapper class
 * 
 * Manages the lifecycle of a WebLLM model:
 * - Loading/unloading models
 * - Single-turn generation
 * - Multi-turn chat conversations
 * - Streaming token generation
 */
class LLMService {
  constructor() {
    this.engine = null;       // WebLLM engine instance
    this.isLoaded = false;    // Model ready state
    this.isLoading = false;   // Loading in progress
    this.currentModel = null; // Currently loaded model ID
  }

  /**
   * Loads a model into the WebLLM engine
   * 
   * Loading process:
   * 1. Check if same model already loaded (skip if so)
   * 2. Unload any existing model
   * 3. Download model weights (cached in IndexedDB)
   * 4. Compile model for WebGPU
   * 5. Initialize inference engine
   * 
   * @param {string} modelId - Model identifier from getAvailableModels()
   * @param {Function} onProgress - Progress callback ({ text, progress })
   */
  async load(modelId, onProgress) {
    // Skip if same model already loaded
    if (this.isLoaded && this.currentModel === modelId) {
      return;
    }
    
    // Unload existing model if switching
    if (this.engine && this.currentModel !== modelId) {
      await this.unload();
    }
    
    this.isLoading = true;
    
    try {
      // Create WebLLM engine with progress tracking
      // This downloads model weights and compiles for WebGPU
      this.engine = await webllm.CreateMLCEngine(modelId, {
        initProgressCallback: (progress) => {
          onProgress?.({
            text: progress.text,
            progress: progress.progress  // 0-1 fraction
          });
        }
      });
      
      this.currentModel = modelId;
      this.isLoaded = true;
    } catch (error) {
      console.error('Failed to load LLM:', error);
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Unloads the current model to free memory
   */
  async unload() {
    if (this.engine) {
      try {
        await this.engine.unload();
      } catch (e) {
        console.warn('Error unloading model:', e);
      }
      this.engine = null;
      this.isLoaded = false;
      this.currentModel = null;
    }
  }

  /**
   * Generates a response to a single prompt
   * 
   * Uses the OpenAI-compatible chat completions API format.
   * Streams tokens as they're generated for responsive UX.
   * 
   * @param {string} prompt - User's input prompt
   * @param {Function} onToken - Callback for streaming (receives full text so far)
   * @returns {Promise<string>} Complete generated response
   */
  async generate(prompt, onToken) {
    if (!this.isLoaded) {
      throw new Error('LLM not loaded');
    }

    // Format as chat messages
    const messages = [
      { role: 'system', content: 'You are a helpful AI assistant. Provide clear, concise responses.' },
      { role: 'user', content: prompt }
    ];

    let fullResponse = '';
    
    // Create streaming completion
    const asyncChunkGenerator = await this.engine.chat.completions.create({
      messages,
      temperature: 0.7,     // Balance creativity/consistency
      max_tokens: 512,      // Reasonable length limit
      stream: true          // Enable streaming
    });

    // Process streamed chunks
    for await (const chunk of asyncChunkGenerator) {
      const delta = chunk.choices[0]?.delta?.content || '';
      fullResponse += delta;
      onToken?.(fullResponse);  // Callback with full text so far
    }

    return fullResponse;
  }

  /**
   * Continues a multi-turn conversation
   * 
   * Maintains conversation context by passing full message history.
   * Each message has a role ('user' or 'assistant') and content.
   * 
   * @param {Array<Object>} messages - Conversation history [{role, content}, ...]
   * @param {Function} onToken - Streaming callback
   * @returns {Promise<string>} Assistant's response
   */
  async chat(messages, onToken) {
    if (!this.isLoaded) {
      throw new Error('LLM not loaded');
    }

    // Prepend system message to conversation
    const formattedMessages = [
      { role: 'system', content: 'You are a helpful AI assistant. Provide clear, concise responses.' },
      ...messages
    ];

    let fullResponse = '';
    
    const asyncChunkGenerator = await this.engine.chat.completions.create({
      messages: formattedMessages,
      temperature: 0.7,
      max_tokens: 512,
      stream: true
    });

    for await (const chunk of asyncChunkGenerator) {
      const delta = chunk.choices[0]?.delta?.content || '';
      fullResponse += delta;
      onToken?.(fullResponse);
    }

    return fullResponse;
  }
}

// Export singleton instance
export const llm = new LLMService();
