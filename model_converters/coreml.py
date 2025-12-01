#!/usr/bin/env python3
"""
CoreML Conversion Script for NVIDIA Prompt Task and Complexity Classifier

This script converts the NVIDIA prompt-task-and-complexity-classifier model from
PyTorch/Hugging Face format to Apple's CoreML format (.mlpackage) for on-device
inference on iOS/macOS.

=== MODEL OVERVIEW ===

Source Model: nvidia/prompt-task-and-complexity-classifier
Architecture: DeBERTa-v3-base with 8 classification heads
Purpose: Classifies prompts by task type and complexity dimensions

Classification Heads (8 outputs):
  0. task_type           - 12 classes (Brainstorming, Code Generation, etc.)
  1. creativity_scope    - 3 classes (High, Medium, Low)
  2. reasoning           - 2 classes (Required, Not Required)
  3. contextual_knowledge- 2 classes (Required, Not Required)
  4. number_of_few_shots - 6 classes (0-5 examples needed)
  5. domain_knowledge    - 4 classes (Expert, Intermediate, Basic, None)
  6. no_label_reason     - 1 class (internal use)
  7. constraint_ct       - 2 classes (Has Constraints, No Constraints)

=== CONVERSION CHALLENGES ===

DeBERTa models use JIT-compiled functions that are incompatible with CoreML's
tracing. This script patches two problematic functions:

1. scaled_size_sqrt: Uses dynamic tensor operations that fail during tracing
2. build_rpos: Has conditional branches with different output shapes

=== OUTPUT ===

The script produces a .mlpackage file that:
- Takes input_ids and attention_mask as inputs (shape: [1, 128])
- Returns 8 separate logit tensors (raw, unnormalized scores)
- Post-processing (softmax, score computation) is done in Swift

=== USAGE ===

Prerequisites:
    pip install torch transformers coremltools safetensors huggingface_hub

Run:
    python coreml.py

Output:
    PromptGrader.mlpackage (copy to Xcode project)

=== POST-PROCESSING (Swift) ===

After getting logits from CoreML, apply:
1. Softmax to convert logits to probabilities
2. Weighted sum using weights_map from model config
3. Divide by divisor_map to normalize scores

See PromptRouter.swift for the complete Swift implementation.

Author: Converted for on-device inference
License: Subject to NVIDIA's model license
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import coremltools
from transformers.models.deberta_v2 import modeling_deberta_v2

# ============================================
# DeBERTa Compatibility Patches
# ============================================
# 
# DeBERTa uses @torch.jit.script decorated functions that cause issues
# during CoreML conversion. We replace them with pure Python equivalents.

def safe_scaled_size_sqrt(query_layer, scale_factor):
    """
    Replacement for DeBERTa's scaled_size_sqrt function.
    
    Original uses dynamic tensor operations that fail during JIT tracing.
    This version explicitly creates a float tensor for the sqrt operation.
    
    Args:
        query_layer: Query tensor from attention layer
        scale_factor: Scaling factor for attention scores
    
    Returns:
        Square root of (query dimension * scale factor)
    """
    return torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)


def safe_build_rpos(query_layer, key_layer, relative_pos, position_buckets, max_relative_positions):
    """
    Replacement for DeBERTa's build_rpos function.
    
    Original has conditional branches that produce different tensor shapes,
    which CoreML's tracer cannot handle. This version evaluates the condition
    at trace time, producing a single code path.
    
    Args:
        query_layer: Query tensor
        key_layer: Key tensor
        relative_pos: Pre-computed relative positions
        position_buckets: Number of position buckets
        max_relative_positions: Maximum relative position value
    
    Returns:
        Relative position tensor for attention computation
    """
    if key_layer.size(-2) != query_layer.size(-2):
        # Cross-attention case: rebuild relative positions
        return modeling_deberta_v2.build_relative_position(
            key_layer,
            key_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    else:
        # Self-attention case: use pre-computed positions
        return relative_pos


# Apply patches to DeBERTa module
modeling_deberta_v2.scaled_size_sqrt = safe_scaled_size_sqrt
modeling_deberta_v2.build_rpos = safe_build_rpos
print("✅ Patched DeBERTa functions for CoreML compatibility.")


# ============================================
# Model Architecture Components
# ============================================

class MeanPooling(nn.Module):
    """
    Mean pooling layer that averages token embeddings.
    
    Takes the last hidden state from the transformer and computes
    a mean representation, weighted by the attention mask to ignore
    padding tokens.
    
    Input:  [batch, seq_len, hidden_dim]
    Output: [batch, hidden_dim]
    """
    
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # Expand mask to match hidden state dimensions
        # [batch, seq_len] -> [batch, seq_len, hidden_dim]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        
        # Sum embeddings, weighted by mask (zeros out padding)
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        
        # Count non-padding tokens
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  # Avoid division by zero
        
        # Compute mean
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    """
    Simple classification head: single linear layer.
    
    Maps pooled representation to class logits.
    
    Input:  [batch, hidden_dim]
    Output: [batch, num_classes]
    """
    
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# ============================================
# Main Model Class
# ============================================

class OriginalCustomModel(nn.Module):
    """
    NVIDIA Prompt Classifier model adapted for CoreML export.
    
    Architecture:
    1. DeBERTa-v3-base backbone (12 layers, 768 hidden dim)
    2. Mean pooling layer
    3. 8 separate classification heads
    
    The forward pass returns raw logits only (no softmax/post-processing).
    Post-processing is handled in Swift for better CoreML compatibility.
    
    Head Output Sizes (from config.target_sizes):
    - task_type: 12 classes
    - creativity_scope: 3 classes
    - reasoning: 2 classes
    - contextual_knowledge: 2 classes
    - number_of_few_shots: 6 classes
    - domain_knowledge: 4 classes
    - no_label_reason: 1 class
    - constraint_ct: 2 classes
    """
    
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(OriginalCustomModel, self).__init__()
        
        # Load DeBERTa backbone
        self.backbone = AutoModel.from_pretrained("microsoft/deberta-v3-base")
        self.target_sizes = target_sizes.values()
        
        # Create classification heads (one per output)
        # Using add_module to match original weight key names (head_0, head_1, etc.)
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz) 
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        
        self.pool = MeanPooling()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass returning raw logits for all 8 heads.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
        
        Returns:
            Tuple of 8 logit tensors, one per classification head
        """
        # Cast mask to float for DeBERTa compatibility
        attention_mask_float = attention_mask.to(dtype=torch.float32)
        
        # Get transformer outputs
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask_float
        )
        last_hidden_state = outputs.last_hidden_state
        
        # Pool to single vector per sequence
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask_float)
        
        # Get logits from each head
        logits = []
        for i in range(len(self.target_sizes)):
            head = getattr(self, f"head_{i}")
            logits.append(head(mean_pooled_representation))
        
        return tuple(logits)


# ============================================
# Main Conversion Script
# ============================================

def main():
    """
    Main conversion pipeline:
    1. Load model configuration from Hugging Face
    2. Initialize model architecture
    3. Download and load pretrained weights
    4. Trace model with sample input
    5. Convert to CoreML format
    6. Save .mlpackage file
    """
    
    model_id = "nvidia/prompt-task-and-complexity-classifier"
    
    # Step 1: Load configuration
    print(f"📋 Loading config for {model_id}...")
    config = AutoConfig.from_pretrained(model_id)
    
    print("   Target sizes:", dict(config.target_sizes))
    print("   Task types:", len(config.task_type_map), "classes")
    
    # Step 2: Download pretrained weights
    print("\n📥 Downloading weights...")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    try:
        # Try safetensors format first (smaller, faster)
        model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        state_dict = load_file(model_file)
        print("   Loaded from model.safetensors")
    except Exception:
        # Fall back to PyTorch bin format
        print("   Safetensors not found, trying pytorch_model.bin...")
        model_file = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")
    
    # Step 3: Initialize model and load weights
    print("\n🔧 Initializing model...")
    model = OriginalCustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    )
    
    # Load weights (strict=False for potential buffer mismatches)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("   Model loaded successfully")
    
    # Step 4: Prepare tokenizer and sample input
    print("\n📝 Preparing sample input...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    sample_text = "Explain the theory of relativity."
    inputs = tokenizer(
        sample_text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    print(f"   Sample: '{sample_text}'")
    print(f"   Input shape: {inputs['input_ids'].shape}")
    
    # Step 5: Trace model with JIT
    print("\n🔍 Tracing model with TorchScript...")
    traced_model = torch.jit.trace(
        model, 
        (inputs["input_ids"], inputs["attention_mask"])
    )
    print("   Tracing successful")
    
    # Step 6: Convert to CoreML
    print("\n🍎 Converting to CoreML...")
    
    # Define output names for each classification head
    output_names = [
        "logits_task_type",            # Head 0: Task classification
        "logits_creativity_scope",     # Head 1: Creativity level
        "logits_reasoning",            # Head 2: Reasoning required
        "logits_contextual_knowledge", # Head 3: Context required
        "logits_few_shots",            # Head 4: Few-shot examples needed
        "logits_domain_knowledge",     # Head 5: Domain expertise needed
        "logits_no_label_reason",      # Head 6: Internal use
        "logits_constraint_ct"         # Head 7: Has constraints
    ]
    
    coreml_outputs = [coremltools.TensorType(name=name) for name in output_names]
    
    mlmodel = coremltools.convert(
        traced_model,
        inputs=[
            coremltools.TensorType(
                name="input_ids", 
                shape=inputs["input_ids"].shape, 
                dtype=int
            ),
            coremltools.TensorType(
                name="attention_mask", 
                shape=inputs["attention_mask"].shape, 
                dtype=int
            )
        ],
        outputs=coreml_outputs,
        minimum_deployment_target=coremltools.target.iOS16  # Requires iOS 16+
    )
    
    # Add metadata
    mlmodel.author = "NVIDIA (converted for on-device inference)"
    mlmodel.short_description = (
        "Prompt Task and Complexity Classifier. "
        "Returns raw logits for 8 classification heads. "
        "Post-processing (softmax, weighted scores) required."
    )
    
    # Step 7: Save the model
    output_filename = "PromptGrader.mlpackage"
    mlmodel.save(output_filename)
    
    print(f"\n✅ Success! Saved '{output_filename}'")
    print("\n📋 Next Steps:")
    print("   1. Copy PromptGrader.mlpackage to your Xcode project")
    print("   2. Copy tokenizer files (tokenizer.json, vocab.txt) to app bundle")
    print("   3. Implement post-processing in Swift (see PromptRouter.swift)")
    print("\n📊 Output Heads:")
    for i, name in enumerate(output_names):
        print(f"   {i}: {name}")


if __name__ == "__main__":
    main()
