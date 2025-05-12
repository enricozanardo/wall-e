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

- Transformer architecture with configurable:
  - Model dimensions
  - Feed-forward network size
  - Number of attention heads
  - Number of layers
- Optimized training on tiny stories dataset
- Consistent model file format (.walle extension)
- Easy-to-use scripts for training and generation
- Support for curriculum learning
- Anti-repetition techniques for better text quality

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

## Examples

### Training a model:

```bash
# Train a small model for best speed
./wall-e1.sh train --size small

# Train a larger model for better quality
./wall-e1.sh train --size medium
```

### Generating text:

```bash
# Basic text generation
./wall-e1.sh generate "Once upon a time"

# Generate more text
./wall-e1.sh generate "In a world where magic" --max-tokens 100

# Use a specific model file
./wall-e1.sh generate "The quick brown fox" --model models/my_custom_model.walle
``` 