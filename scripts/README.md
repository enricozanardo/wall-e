# Wall-E1 Model Scripts

This directory contains scripts for training, managing, and generating text with Wall-E1 language models.

## Main Script

The `wall-e1-model.sh` script provides a unified interface for all Wall-E1 model operations:

```bash
# Train a new model
./wall-e1-model.sh train --size small|medium|large [--stories <number>] [--memory-opt] [--batch-size <number>]

# Generate text from a prompt
./wall-e1-model.sh generate "Your prompt here" --max-tokens 50

# Clean all model files
./wall-e1-model.sh clean

# Show help
./wall-e1-model.sh help
```

## Individual Scripts

- **train_optimized_accuracy.sh**: Trains a model with a focus on accuracy.
- **test_generation.sh**: Tests text generation with a trained model.
- **generate_text.py**: Python script for nicely formatted text generation.
- **create_test_model.sh**: Creates a test binary model (mainly for development).
- **profile_memory.sh**: Profiles memory access patterns and bandwidth usage.
- **benchmark_memory.sh**: Benchmarks different memory optimization configurations.

## File Format

All trained models use the `.walle` extension for consistency. The internal format can be either JSON or binary, and the loader will automatically detect the correct format.

## Examples

### Training a model:

```bash
# Train using the main script with default settings
./wall-e1-model.sh train --size small

# Train with custom number of stories
./wall-e1-model.sh train --size medium --stories 2000

# Train with memory optimization
./wall-e1-model.sh train --size medium --memory-opt

# Train with all options
./wall-e1-model.sh train --size large --stories 5000 --cpus 8 --memory-opt --batch-size 128

# Or use the training script directly
./train_optimized_accuracy.sh --size medium
./train_optimized_accuracy.sh --size large --stories 5000 --memory-opt
```

### Generating text:

```bash
# Generate using the main script
./wall-e1-model.sh generate "Once upon a time"

# Or use the generation scripts directly
./test_generation.sh "Once upon a time" 50 models/high_accuracy_model.walle
./generate_text.py "Once upon a time" --max-tokens 100
```

## Memory Optimization

The memory optimization features help overcome memory bandwidth bottlenecks that can limit CPU utilization in deep learning workloads:

- **--memory-opt**: Enables cache-efficient tensor operations and memory bandwidth-based thread allocation
- **--batch-size <number>**: Manually sets a batch size, overriding the automatic cache-optimal calculation

You can use the profiling and benchmarking scripts to measure performance improvements:

```bash
# Profile memory access patterns
./scripts/profile_memory.sh --size medium --detailed

# Benchmark different optimization configurations
./scripts/benchmark_memory.sh
```

## Notes

- All models are saved in the `models/` directory.
- The default model path is `models/high_accuracy_model.walle`.
- Training uses the `tiny_stories_sample.json` dataset by default.
- By default, training uses 4000 stories from the dataset, but this can be customized with the `--stories` parameter.
- Memory optimization provides better performance on machines with limited memory bandwidth. 