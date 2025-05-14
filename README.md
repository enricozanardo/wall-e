# Wall-E1 Language Model

Wall-E1 is a Transformer-based language model implementation in Rust focusing on effective text generation.

## Quick Start

Use the convenience script in the root directory:

```bash
# Train a new model
./wall-e1.sh train --size small

# Generate text from a prompt
./wall-e1.sh generate "Once upon a time"

# Clean all model files
./wall-e1.sh clean
```

## Features

- High-performance, memory-efficient Rust implementation
- Transformer-based architecture with self-attention
- Optimized tokenization with WordPiece BPE algorithm
- Curriculum learning for progressive model training
- Support for both JSON and binary model formats
- Text generation with anti-repetition mechanisms

## Model Sizes

| Size   | Dimensions | FF Size | Heads | Layers |
|--------|------------|---------|-------|--------|
| Small  | 128        | 512     | 4     | 3      |
| Medium | 256        | 1024    | 8     | 6      |
| Large  | 512        | 2048    | 8     | 8      |

## Training Data

The model is trained on the `tiny_stories_sample.json` dataset, which contains short stories that are ideal for training language models.

## Directory Structure

- `src/` - Source code for the Wall-E1 implementation
- `scripts/` - Utility scripts for training and text generation
- `models/` - Directory where trained models are stored
- `data/` - Training data files

## Advanced Usage

See the README in the scripts directory for more detailed usage instructions:

```bash
# View scripts documentation
cat scripts/README.md
```

## Memory Optimization

Wall-E1 includes advanced memory optimization techniques to address bandwidth limitations and improve performance:

- **Cache-efficient operations**: Matrix multiplication and tensor operations are optimized for CPU cache utilization
- **Memory-aware batch sizing**: Automatically determines optimal batch sizes based on available cache
- **Hardware detection**: Detects CPU cache parameters to tune algorithm performance
- **Prefetching**: Uses strategic data prefetching for improved memory access patterns
- **Blocked algorithms**: Implements cache-blocked matrix multiplication for better spatial locality
- **Thread optimization**: Intelligently allocates threads based on memory bandwidth capabilities

## Usage

### Training a Model

```bash
# Basic training with default parameters
./scripts/wall-e1-model.sh train --size small

# Training with memory optimization
./scripts/wall-e1-model.sh train --size medium --memory-opt

# Advanced options
./scripts/wall-e1-model.sh train --size large --stories 5000 --cpus 8 --memory-opt --batch-size 128
```

### Generating Text

```bash
# Generate text from default model
./scripts/wall-e1-model.sh generate "Once upon a time"

# Generate text with custom parameters
./scripts/wall-e1-model.sh generate "Hello world" --model models/my_model.walle --max-tokens 100
```

## Command Line Options

### Training Options

- `--size [small|medium|large]`: Set model size (default: small)
- `--stories [number]`: Number of stories to use for training (default: 4000)
- `--cpus [number]`: Number of CPU cores to use (default: all available)
- `--memory-opt`: Enable memory optimization for better cache utilization and thread allocation
- `--batch-size [number]`: Manually set batch size (overrides automatic calculation)

### Generation Options

- `--model [path]`: Model file path (default: models/high_accuracy_model.walle)
- `--max-tokens [number]`: Maximum tokens to generate (default: 50)
- `--cpus [number]`: Number of CPU cores to use (default: all available)

## Architecture

Wall-E1 uses a Transformer-based architecture with the following components:

- Multi-head self-attention mechanism
- Feed-forward neural networks with ReLU activation
- Layer normalization and residual connections
- Positional encoding for sequence awareness
- Curriculum learning for improved training efficiency
- Memory optimization for cache-efficient operations

## Model Formats

Wall-E1 supports both JSON and binary model formats with the `.walle` extension. The format is automatically detected during loading.

## Development

Wall-E1 is written in Rust and requires Rust 1.65 or later. Key components:

- `nabla`: Tensor operations and automatic differentiation
- `memory_opt`: Memory optimization and cache-efficient algorithms
- `tokenizer`: WordPiece BPE tokenization
- `training`: Model training and evaluation
- `export`: Model serialization and loading 