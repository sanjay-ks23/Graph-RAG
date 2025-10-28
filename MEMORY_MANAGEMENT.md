# CUDA Memory Management Guide

This guide explains how to handle CUDA out-of-memory errors in the Graph RAG system.

## What Was Fixed

The system now includes automatic CUDA cache clearing at critical points:

1. **Before loading models** (Embedding & LLM)
2. **Before batch operations** (embedding generation)
3. **After generation** (response generation)
4. **At pipeline stages** (indexing)

## Configuration Changes

### 1. Environment Variable (Recommended)

The `.env` file now includes:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This reduces memory fragmentation. Make sure to copy from `.env.example`:
```bash
cp .env.example .env
```

### 2. Reduced Batch Size

In `config.yaml`, the embedding batch size was reduced:
```yaml
embedding:
  batch_size: 16  # Reduced from 32
```

You can reduce this further if needed:
```yaml
embedding:
  batch_size: 8  # For 4GB GPUs
  batch_size: 4  # For 2GB GPUs
```

## Running with Limited GPU Memory

### Option 1: Use Smaller Batch Sizes

Edit `config.yaml`:
```yaml
models:
  embedding:
    batch_size: 8  # Adjust based on your GPU
```

### Option 2: Use CPU for Embedding Model

If you have limited GPU memory, use CPU for embeddings and GPU for LLM:

```yaml
models:
  llm:
    device: "cuda"  # Keep LLM on GPU (most important)
  
  embedding:
    device: "cpu"   # Move embeddings to CPU
    batch_size: 32  # Can increase on CPU
```

### Option 3: Use CPU for Everything

Edit `config.yaml`:
```yaml
models:
  llm:
    device: "cpu"
  
  embedding:
    device: "cpu"
```

**Note**: This will be slower but won't have memory issues.

## Manual Cache Clearing

You can manually clear CUDA cache in Python:

```python
import torch
torch.cuda.empty_cache()
```

Or use the utility function:
```python
from src.utils.memory_utils import clear_cuda_cache
clear_cuda_cache()
```

## Monitoring GPU Memory

### During Indexing

The pipeline now logs GPU memory at key stages:
```
GPU Memory Status:
  Total: 7.63 GB
  Allocated: 4.98 GB
  Reserved: 1.23 GB
  Free: 2.65 GB
```

### Manual Check

```python
from src.utils.memory_utils import log_gpu_memory
log_gpu_memory()
```

Or use PyTorch directly:
```python
import torch
print(torch.cuda.memory_summary())
```

## Troubleshooting

### Error: "CUDA out of memory"

**Solutions (in order of preference):**

1. **Set environment variable** (already done in `.env.example`):
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

2. **Reduce batch size** in `config.yaml`:
   ```yaml
   embedding:
     batch_size: 8  # or 4
   ```

3. **Move embedding to CPU**:
   ```yaml
   embedding:
     device: "cpu"
   ```

4. **Use CPU for everything**:
   ```yaml
   llm:
     device: "cpu"
   embedding:
     device: "cpu"
   ```

5. **Close other GPU applications**:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Kill other processes if needed
   ```

### Error: "CUDA device-side assert triggered"

This usually means a different issue. Check:
- Model compatibility
- Input tensor shapes
- Token IDs within vocabulary range

### Slow Performance on CPU

If using CPU mode is too slow:
1. Reduce the number of documents
2. Use smaller chunk sizes
3. Reduce retrieval top-k values

## Best Practices

### For 8GB GPU (like yours - 7.63 GB)

```yaml
models:
  llm:
    device: "cuda"
  
  embedding:
    device: "cuda"
    batch_size: 8  # Conservative
```

### For 4GB GPU

```yaml
models:
  llm:
    device: "cuda"
  
  embedding:
    device: "cpu"  # Move to CPU
    batch_size: 32
```

### For No GPU

```yaml
models:
  llm:
    device: "cpu"
  
  embedding:
    device: "cpu"
```

## Memory-Efficient Tips

1. **Process documents in batches**: Don't load all books at once
2. **Clear cache regularly**: Already implemented in the code
3. **Use FP16**: Already enabled for CUDA
4. **Enable TF32**: Already enabled for Ampere GPUs
5. **Monitor memory**: Use `log_gpu_memory()` frequently

## Running the System

### With Current Settings (8GB GPU)

```bash
# Make sure .env is set up
cp .env.example .env

# Run indexing
python index_documents.py

# Run app
python app.py
```

### If You Still Get OOM Errors

```bash
# Edit config.yaml and reduce batch_size to 8 or 4
nano config.yaml

# Then run again
python index_documents.py
```

## Code Changes Made

All these files now include automatic cache clearing:

- `src/embedding/embedding_model.py` - Clears before/after encoding
- `src/llm/gemma_model.py` - Clears before/after generation
- `src/pipeline/indexing_pipeline.py` - Clears at pipeline stages
- `src/utils/memory_utils.py` - Utility functions for memory management
- `app.py` - Clears at initialization

## Summary

The system now automatically manages CUDA memory. If you still encounter issues:

1. ✅ Environment variable is set (`.env`)
2. ✅ Batch size is reduced (config.yaml)
3. ✅ Cache clearing is automatic (code)
4. 🔧 Adjust batch_size further if needed
5. 🔧 Move embedding to CPU if necessary

Your 7.63 GB GPU should work fine with `batch_size: 8-16` for embeddings.
