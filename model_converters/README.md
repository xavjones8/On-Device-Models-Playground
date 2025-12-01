# Model Converters

Python scripts to convert the NVIDIA prompt-task-and-complexity-classifier model to different formats for on-device inference.

## Source Model

- **Model**: `nvidia/prompt-task-and-complexity-classifier`
- **Architecture**: DeBERTa-v3-base backbone with multi-head classification
- **Outputs**: 8 classification heads (task type + 7 complexity dimensions)

## Scripts

### `coreml.py` - CoreML Conversion (iOS/macOS)

Converts the model to Apple's CoreML format (`.mlpackage`).

```bash
# Setup
pip install torch transformers coremltools safetensors huggingface_hub

# Run
python coreml.py
```

**Output**: `PromptGrader.mlpackage`

**Key Challenges Solved**:
- DeBERTaV2 JIT compilation issues (patched `scaled_size_sqrt`, `build_rpos`)
- `torch.sqrt` int32 input handling
- Model outputs raw logits (post-processing in Swift)

### `onnx_convert.py` - ONNX Conversion (Web)

Converts the model to ONNX format for browser inference.

```bash
# Setup
pip install torch transformers onnx onnxruntime safetensors huggingface_hub

# Run
python onnx_convert.py
```

**Output Directory**: `../onnx_model/`
- `model.onnx` - The converted model
- `tokenizer.json` - Tokenizer vocabulary
- `classifier_metadata.json` - Weight maps, divisors, task type labels

## Model Architecture

```
Input: input_ids, attention_mask (max_length=512)
         │
         ▼
┌─────────────────────┐
│  DeBERTa-v3-base    │
│  (Backbone)         │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Mean Pooling      │
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              8 Classification Heads                      │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ task_type│creativity│reasoning │contextual│ few_shots   │
│ (12 cls) │ (3 cls)  │ (2 cls)  │ (2 cls)  │ (6 cls)     │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│ domain   │no_label  │constraint│          │             │
│ (4 cls)  │(2 cls)   │ (6 cls)  │          │             │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
```

## Post-Processing (Done in Swift/JavaScript)

1. **Task Type**: Softmax → Top-2 predictions with probabilities
2. **Complexity Scores**: Softmax → Weighted sum → Normalize by divisor
3. **Aggregate Score**: Weighted combination of all complexity dimensions

```
aggregate = 0.35 * creativity
          + 0.25 * reasoning
          + 0.15 * constraint
          + 0.15 * domain_knowledge
          + 0.05 * contextual_knowledge
          + 0.05 * few_shots
```

## Troubleshooting

### CoreML: "unsupported op" errors
- Ensure patches are applied before model loading
- Check CoreML version compatibility

### ONNX: Slow export
- Use `dynamo=False` for faster legacy export
- Verify ONNX Runtime version matches

### Both: Incorrect outputs
- Verify weight maps and divisor maps match config.json
- Check tokenizer max_length (512)
- Ensure padding/truncation is consistent


