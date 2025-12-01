#!/usr/bin/env python3
"""
ONNX Conversion Script for NVIDIA Prompt Task and Complexity Classifier

This script converts the NVIDIA prompt-task-and-complexity-classifier model from
PyTorch/Hugging Face format to ONNX format for browser-based inference using
ONNX Runtime Web.

=== MODEL OVERVIEW ===

Source Model: nvidia/prompt-task-and-complexity-classifier
Architecture: DeBERTa-v3-base with 8 classification heads
Purpose: Classifies prompts by task type and complexity dimensions

Classification Heads (8 outputs):
  0. logits_task_type           - 12 classes (Brainstorming, Code Gen, etc.)
  1. logits_creativity_scope    - 3 classes (High, Medium, Low)
  2. logits_reasoning           - 2 classes (Required, Not Required)
  3. logits_contextual_knowledge- 2 classes (Required, Not Required)
  4. logits_few_shots           - 6 classes (0-5 examples needed)
  5. logits_domain_knowledge    - 4 classes (Expert, Intermediate, Basic, None)
  6. logits_no_label_reason     - 1 class (internal use)
  7. logits_constraint_ct       - 2 classes (Has Constraints, No Constraints)

=== OUTPUT FILES ===

The script produces the following files in ./onnx_model/:

  model.onnx              - The ONNX model (~700MB)
  tokenizer.json          - Tokenizer vocabulary and config
  tokenizer_config.json   - Tokenizer settings
  special_tokens_map.json - Special token mappings
  config.json             - Model configuration
  classifier_metadata.json- Weights/divisors for score computation

=== USAGE ===

Prerequisites:
    pip install torch transformers onnx onnxruntime safetensors huggingface_hub

Run:
    python onnx_convert.py

Output:
    ./onnx_model/  (copy contents to web-app/public/models/)

=== WEB DEPLOYMENT ===

1. Copy model.onnx to your web app's public/models/ directory
2. Copy tokenizer files for @huggingface/transformers
3. Copy classifier_metadata.json for post-processing weights
4. Use ONNX Runtime Web for inference (see classifier.js)

=== POST-PROCESSING (JavaScript) ===

After getting logits from ONNX Runtime:
1. Apply softmax to convert logits to probabilities
2. Compute weighted sum using weights_map
3. Divide by divisor_map to normalize scores

See web-app/src/classifier.js for the complete implementation.

Author: Converted for browser-based inference
License: Subject to NVIDIA's model license
"""

import os
import json
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


# ============================================
# Model Architecture Components
# ============================================

class MeanPooling(nn.Module):
    """
    Mean pooling layer that averages token embeddings.
    
    Computes a weighted mean of the last hidden state, where weights
    are determined by the attention mask (ignoring padding tokens).
    
    Input:  last_hidden_state [batch, seq_len, hidden_dim]
            attention_mask [batch, seq_len]
    Output: [batch, hidden_dim]
    """
    
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand attention mask to hidden dimension
        # Shape: [batch, seq_len] -> [batch, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        
        # Weighted sum of embeddings (padding tokens contribute 0)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        # Count non-padding tokens per sequence
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Prevent division by zero
        
        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    """
    Classification head: single linear layer mapping pooled features to logits.
    
    Input:  [batch, hidden_dim]
    Output: [batch, num_classes]
    """
    
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class PromptClassifierONNX(nn.Module):
    """
    ONNX-optimized version of the NVIDIA Prompt Classifier.
    
    This class restructures the original model for clean ONNX export:
    - Uses nn.ModuleList instead of add_module for cleaner weight mapping
    - Returns raw logits as a tuple (ONNX supports multiple outputs)
    - No post-processing (softmax/scoring done in JavaScript)
    
    Architecture:
        DeBERTa-v3-base backbone
            ↓
        Mean Pooling [batch, 768]
            ↓
        8 Classification Heads → 8 Logit Tensors
    """
    
    def __init__(self, config):
        super(PromptClassifierONNX, self).__init__()
        
        # DeBERTa-v3-base: 12 layers, 768 hidden dim, 12 attention heads
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        
        # Get output sizes for each head from config
        # e.g., {'task_type': 12, 'creativity_scope': 3, ...}
        self.target_sizes = list(config.target_sizes.values())
        
        # Create classification heads using ModuleList
        # This produces clean weight keys: heads.0.fc.weight, heads.1.fc.weight, etc.
        self.heads = nn.ModuleList([
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ])
        
        self.pool = MeanPooling()
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass returning raw logits for all classification heads.
        
        Args:
            input_ids: Token IDs [batch, seq_len] (int64)
            attention_mask: Attention mask [batch, seq_len] (int64)
        
        Returns:
            Tuple of 8 logit tensors:
                - logits_task_type [batch, 12]
                - logits_creativity_scope [batch, 3]
                - logits_reasoning [batch, 2]
                - logits_contextual_knowledge [batch, 2]
                - logits_few_shots [batch, 6]
                - logits_domain_knowledge [batch, 4]
                - logits_no_label_reason [batch, 1]
                - logits_constraint_ct [batch, 2]
        """
        # Cast attention mask to float for DeBERTa's attention computation
        attention_mask_float = attention_mask.float()
        
        # Get transformer hidden states
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask_float
        )
        last_hidden_state = outputs.last_hidden_state
        
        # Pool sequence to single vector
        pooled = self.pool(last_hidden_state, attention_mask_float)
        
        # Get logits from each classification head
        logits = [head(pooled) for head in self.heads]
        
        return tuple(logits)


def load_weights_from_original(model, state_dict):
    """
    Maps weights from original model format to ONNX-friendly format.
    
    The original NVIDIA model uses add_module which creates keys like:
        head_0.fc.weight, head_0.fc.bias, head_1.fc.weight, ...
    
    Our ModuleList-based model expects:
        heads.0.fc.weight, heads.0.fc.bias, heads.1.fc.weight, ...
    
    This function renames the keys to match our structure.
    
    Args:
        model: PromptClassifierONNX instance
        state_dict: Original model weights
    
    Returns:
        Model with loaded weights
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("head_"):
            # head_0.fc.weight -> heads.0.fc.weight
            new_key = key.replace("head_", "heads.")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle any buffer mismatches
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if missing:
        print(f"   ⚠️ Missing keys (may be expected): {len(missing)} keys")
    if unexpected:
        print(f"   ⚠️ Unexpected keys: {unexpected}")
    
    return model


def main():
    """
    Main conversion pipeline:
    
    1. Load model configuration from Hugging Face Hub
    2. Initialize ONNX-friendly model architecture
    3. Download and load pretrained weights
    4. Test forward pass
    5. Export to ONNX format
    6. Verify ONNX model
    7. Test with ONNX Runtime
    8. Save tokenizer and metadata files
    """
    
    model_id = "nvidia/prompt-task-and-complexity-classifier"
    output_dir = "./onnx_model"
    
    # ==========================================
    # Step 1: Load Configuration
    # ==========================================
    print(f"📋 Loading config from {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    
    print("\n📊 Model configuration:")
    print(f"   Target sizes: {config.target_sizes}")
    print(f"   Task types: {len(config.task_type_map)} classes")
    print(f"   Task type map: {config.task_type_map}")
    
    # ==========================================
    # Step 2: Initialize Model
    # ==========================================
    print("\n🔧 Initializing ONNX-friendly model...")
    model = PromptClassifierONNX(config)
    
    # ==========================================
    # Step 3: Download and Load Weights
    # ==========================================
    print("\n📥 Downloading pretrained weights...")
    try:
        # Prefer safetensors format (faster loading, smaller files)
        model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(model_file)
        print("   Loaded from model.safetensors")
    except Exception:
        # Fall back to PyTorch binary format
        print("   Safetensors not found, trying pytorch_model.bin...")
        model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")
    
    print("🔄 Loading weights into model...")
    model = load_weights_from_original(model, state_dict)
    model.eval()  # Set to evaluation mode (disables dropout)
    
    # ==========================================
    # Step 4: Prepare Tokenizer and Test Input
    # ==========================================
    print("\n📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Create dummy input for tracing and testing
    print("🔄 Creating test input...")
    dummy_text = "Write a Python function to sort a list."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )
    print(f"   Test prompt: '{dummy_text}'")
    print(f"   Input shape: {inputs['input_ids'].shape}")
    
    # ==========================================
    # Step 5: Test Forward Pass
    # ==========================================
    print("\n🧪 Testing PyTorch forward pass...")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        print(f"   ✅ Forward pass successful")
        print(f"   Output shapes: {[o.shape for o in outputs]}")
    
    # ==========================================
    # Step 6: Export to ONNX
    # ==========================================
    print(f"\n🔄 Exporting to ONNX format...")
    os.makedirs(output_dir, exist_ok=True)
    
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    # Define input/output names (must match what we use in JavaScript)
    input_names = ["input_ids", "attention_mask"]
    output_names = [
        "logits_task_type",           # 12 classes
        "logits_creativity_scope",    # 3 classes
        "logits_reasoning",           # 2 classes
        "logits_contextual_knowledge",# 2 classes
        "logits_few_shots",           # 6 classes
        "logits_domain_knowledge",    # 4 classes
        "logits_no_label_reason",     # 1 class
        "logits_constraint_ct"        # 2 classes
    ]
    
    # Dynamic axes allow variable batch size and sequence length
    # This enables batched inference and different prompt lengths
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    }
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}
    
    # Export using legacy TorchScript-based exporter
    # (dynamo=False is faster and more stable for transformer models)
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,           # ONNX opset 14 for broad compatibility
        do_constant_folding=True,   # Optimize constant expressions
        dynamo=False,               # Use legacy exporter (faster, more stable)
    )
    
    print(f"   ✅ ONNX model saved to: {onnx_path}")
    
    # ==========================================
    # Step 7: Verify ONNX Model
    # ==========================================
    print("\n🔍 Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("   ✅ ONNX model is valid!")
    
    # ==========================================
    # Step 8: Test with ONNX Runtime
    # ==========================================
    print("\n🧪 Testing with ONNX Runtime...")
    import onnxruntime as ort
    
    # Create inference session
    session = ort.InferenceSession(onnx_path)
    
    # Prepare inputs (ONNX Runtime expects numpy arrays)
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }
    
    # Run inference
    ort_outputs = session.run(None, ort_inputs)
    
    print("   ✅ ONNX Runtime inference successful!")
    print(f"   Output shapes: {[o.shape for o in ort_outputs]}")
    
    # Compare PyTorch vs ONNX outputs (should be nearly identical)
    print("\n📊 Comparing PyTorch vs ONNX outputs...")
    for i, (pt_out, ort_out) in enumerate(zip(outputs, ort_outputs)):
        diff = abs(pt_out.numpy() - ort_out).max()
        status = "✅" if diff < 1e-4 else "⚠️"
        print(f"   {status} {output_names[i]}: max diff = {diff:.6f}")
    
    # ==========================================
    # Step 9: Save Tokenizer Files
    # ==========================================
    print("\n💾 Saving tokenizer files...")
    tokenizer.save_pretrained(output_dir)
    print(f"   Saved to {output_dir}/")
    
    # ==========================================
    # Step 10: Save Model Config
    # ==========================================
    print("💾 Saving model config...")
    config.save_pretrained(output_dir)
    
    # ==========================================
    # Step 11: Create Metadata File for Web App
    # ==========================================
    print("💾 Creating classifier metadata...")
    
    # This metadata is needed for post-processing in JavaScript
    metadata = {
        "task_type_map": config.task_type_map,      # Index -> Task name
        "weights_map": config.weights_map,          # Score computation weights
        "divisor_map": config.divisor_map,          # Score normalization divisors
        "target_sizes": config.target_sizes,        # Output sizes per head
        "output_names": output_names                # ONNX output tensor names
    }
    
    metadata_path = os.path.join(output_dir, "classifier_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Saved to {metadata_path}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*50)
    print("📦 Output Files:")
    print("="*50)
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   {filename}: {size_mb:.2f} MB")
    
    print("\n✅ Conversion complete!")
    print(f"\n📋 Next Steps:")
    print(f"   1. Copy {output_dir}/model.onnx to web-app/public/models/")
    print(f"   2. Copy tokenizer files to web-app/public/models/")
    print(f"   3. Copy classifier_metadata.json to web-app/public/models/")
    print(f"   4. Update classifier.js to load from these paths")
    print(f"\n📊 Output Heads:")
    for i, name in enumerate(output_names):
        size = list(config.target_sizes.values())[i]
        print(f"   {i}: {name} ({size} classes)")


if __name__ == "__main__":
    main()
