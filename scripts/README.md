# Wall-E1 Model Scripts

This directory contains scripts for training, managing, and generating text with Wall-E1 language models.

## Main Script

The `wall-e1-model.sh` script provides a unified interface for all Wall-E1 model operations:

```bash
# Train a new model
./wall-e1-model.sh train --size small|medium|large [--stories <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>] [--auto-resize-vocab] [--watchdog-timeout <seconds>] [--data-threads <number>] [--target-id-max <number>] [--vocab-size <number>] [--min-freq <number>] [--enable-skip] [--strong-anti-rep] [--json-format] [--profile|--perf-log]

# Generate text from a prompt
./wall-e1-model.sh generate "Your prompt here" --max-tokens 50 [--disable-watchdog]

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

# Train with specific gradient checkpointing strategy
./wall-e1-model.sh train --size medium --memory-opt --checkpoint-strategy uniform

# Train with thread optimization for matrix multiplication
./wall-e1-model.sh train --size medium --thread-opt matrix_multiply

# Train with custom number of epochs
./wall-e1-model.sh train --size small --epochs 5

# Train with more curriculum examples to prevent stalling in level advancement
./wall-e1-model.sh train --size large --curriculum-examples 5000

# Train with automatic vocabulary resizing to handle out-of-range target IDs
./wall-e1-model.sh train --size medium --auto-resize-vocab --target-id-max 10000

# Train with custom watchdog and data thread settings for better deadlock prevention
./wall-e1-model.sh train --size small --watchdog-timeout 120 --data-threads 16

# Train with custom vocabulary settings
./wall-e1-model.sh train --size medium --vocab-size 5000 --min-freq 2

# Train with skip connections and anti-repetition for better model architecture
./wall-e1-model.sh train --size medium --enable-skip --strong-anti-rep

# Train with JSON format data source
./wall-e1-model.sh train --size small --json-format --stories 2000

# Train with all options
./wall-e1-model.sh train --size large --stories 5000 --cpus 8 --memory-opt --checkpoint-strategy adaptive --thread-opt gradient_update --batch-size 128 --epochs 15 --curriculum-examples 5000 --auto-resize-vocab --target-id-max 10000 --watchdog-timeout 180 --data-threads 12 --vocab-size 5000 --min-freq 2 --enable-skip --strong-anti-rep --json-format

# Train with performance profiling
./wall-e1-model.sh train --size small --perf-log
./wall-e1-model.sh train --size medium --memory-opt --perf-log

# Or use the training script directly
./train_optimized_accuracy.sh --size medium
./train_optimized_accuracy.sh --size large --stories 5000 --memory-opt
./train_optimized_accuracy.sh --size small --epochs 20
```

### Generating text:

```bash
# Generate using the main script
./wall-e1-model.sh generate "Once upon a time"

# Generate with watchdog disabled to prevent stalled progress warnings
./wall-e1-model.sh generate "Once upon a time" --disable-watchdog

# Or use the generation scripts directly
./test_generation.sh "Once upon a time" 50 models/high_accuracy_model.walle
./generate_text.py "Once upon a time" --max-tokens 100
```

## Memory Optimization

The memory optimization features help overcome memory bandwidth bottlenecks that can limit CPU utilization in deep learning workloads:

- **--memory-opt**: Enables cache-efficient tensor operations, gradient checkpointing, and memory bandwidth-based thread allocation
- **--checkpoint-strategy**: Controls how gradient checkpointing saves memory during backpropagation:
  - `boundary`: Only checkpoint the input/output of layer blocks (minimal memory usage but more recomputation)
  - `uniform`: Checkpoint at regular intervals throughout the network (balanced approach)
  - `adaptive`: Dynamically adjust checkpoints based on memory usage (recommended default)
- **--thread-opt**: Optimizes thread allocation for specific operation types:
  - `matrix_multiply`: For compute-intensive matrix operations
  - `attention`: For attention mechanism operations with heavy memory access patterns
  - `gradient_update`: For applying gradients during optimizer steps
  - `data_loading`: For I/O bound operations like data preprocessing
- **--batch-size <number>**: Manually sets a batch size, overriding the automatic cache-optimal calculation

You can use the profiling and benchmarking scripts to measure performance improvements:

```bash
# Profile memory access patterns
./scripts/profile_memory.sh --size medium --detailed

# Benchmark different optimization configurations
./scripts/benchmark_memory.sh
```

## Training Parameters

- **--size small|medium|large**: Sets the model size, affecting the model dimensions and architecture
- **--stories <number>**: Number of stories to use from the dataset (default: 4000)
- **--cpus <number>**: Number of CPU cores to use for training (default: all available)
- **--memory-opt**: Enable memory optimization for better performance
- **--checkpoint-strategy <strategy>**: Gradient checkpointing strategy (boundary, uniform, adaptive)
- **--thread-opt <operation>**: Optimize thread allocation for specific operation type
- **--batch-size <number>**: Manually set batch size for training
- **--epochs <number>**: Number of training epochs to run (default: 10)
- **--curriculum-examples <number>**: Number of examples to use for curriculum initialization (default: 500). Increasing this value can prevent stalling in curriculum level advancement, especially with larger batch sizes.
- **--auto-resize-vocab**: Automatically resize vocabulary when target IDs exceed the current maximum, preventing "target_id out of range" errors
- **--watchdog-timeout <seconds>**: Set timeout for watchdog thread detection (default: 60). Increase for larger models or slower systems.
- **--data-threads <number>**: Number of threads for data loading (min: 8, default: 70% of available cores). Increasing can help with CPU utilization.
- **--target-id-max <number>**: Maximum target ID value (default: auto-detected, min: 5000). Set higher for larger vocabularies.
- **--vocab-size <number>**: Size of the vocabulary (default: 10000). Smaller values create a more compact model, larger values improve accuracy but increase memory usage.
- **--min-freq <number>**: Minimum token frequency for vocabulary inclusion (default: 2). Higher values create a more focused vocabulary but may increase out-of-vocabulary tokens.
- **--enable-skip**: Enable skip connections in the model architecture, which can improve gradient flow and model performance.
- **--strong-anti-rep**: Enable stronger anti-repetition mechanisms to prevent the model from generating repetitive text.
- **--json-format**: Use JSON format for input data instead of plain text. Required when training on TinyStories or similar JSON-formatted datasets.
- **--profile, --perf-log [true|false]**: Enable detailed performance profiling and metrics collection. You can use it as a flag (--perf-log) or with an explicit value (--perf-log true)

## Thread Pool Optimization

The thread pool optimization features dynamically adjust parallelism based on workload characteristics:

1. **Hardware-aware thread allocation**: Analyzes your CPU topology (physical vs. logical cores) and memory bandwidth to determine optimal thread counts
2. **Workload-specific parallelism**: Different operation types have different optimal thread counts:
   - Matrix multiplication is compute-intensive and benefits from more threads
   - Data loading is I/O bound and uses fewer threads to avoid contention
   - Gradient updates are memory-bandwidth intensive and need careful thread balancing
3. **Dynamic adjustment**: Thread counts are adjusted during training as operations change
4. **Small model handling**: Prevents over-parallelization of small workloads that would create more overhead than benefit

## Gradient Checkpointing

Gradient checkpointing trades computation for memory by selectively saving activations during the forward pass:

1. **Boundary strategy**: Only saves the input and output of layer blocks, minimizing memory usage but requiring more recomputation
2. **Uniform strategy**: Saves activations at regular intervals, providing a balanced approach
3. **Adaptive strategy**: Dynamically adjusts which activations to save based on memory usage patterns
4. **Memory tracking**: Monitors peak memory usage and provides statistics on memory savings

This technique can reduce memory usage by 30-70% with only a 20-30% increase in computation time.

## Thread Deadlock Prevention

Wall-E1 includes several mechanisms to prevent thread deadlocks during training:

1. **Watchdog Thread**: Monitors all worker threads and detects when progress stalls. Set timeout with `--watchdog-timeout`.
2. **Thread State Tracking**: Tracks the state of each thread to provide detailed diagnostics when issues occur.
3. **Recovery Mechanisms**: Automatically attempts to recover from deadlocks by terminating stuck threads.
4. **Optimized Data Threads**: Uses a separate thread pool for data loading operations with `--data-threads`.

If you encounter "Progress stalled" warnings during text generation, use the `--disable-watchdog` option.

## Target ID Handling

When training with large vocabularies, you may encounter "target_id XXX out of range (max YYY)" warnings, which indicate that token IDs in your training data exceed the maximum vocabulary size:

1. **Auto-resize**: Enable `--auto-resize-vocab` to automatically resize the vocabulary when out-of-range tokens are encountered.
2. **Target ID Maximum**: Set `--target-id-max` to pre-allocate a larger vocabulary size.
3. **Monitoring**: The training will report vocabulary usage statistics to help you tune these parameters.

For optimal performance, set `--auto-resize-vocab` with a reasonable `--target-id-max` value based on your dataset size.

## Model Architecture Options

Wall-E1 supports several architecture enhancements that can improve model performance:

1. **Skip Connections**: Enable with `--enable-skip` to add residual connections between layers, which helps gradient flow and can improve training stability and model quality.
2. **Anti-repetition**: Use `--strong-anti-rep` to enable more aggressive penalties for repetitive text generation, which helps prevent common issues like repeated phrases or patterns.
3. **Vocabulary Size**: Adjust with `--vocab-size` to balance between model size and vocabulary coverage. Smaller values create more compact models, while larger values can improve accuracy at the cost of increased memory usage.
4. **Token Frequency**: Use `--min-freq` to control which tokens get included in the vocabulary based on their frequency in the training data. Higher values create a more focused vocabulary.

## Input Data Format

Wall-E1 supports multiple input formats for training data:

1. **Plain Text**: The default format, where the input file contains raw text.
2. **JSON Format**: Enable with `--json-format` for structured datasets like TinyStories, where stories are contained within a JSON structure.

When using JSON format, you can control the number of stories to use with the `--stories` parameter, which is particularly useful for experiments with varying dataset sizes.

## Notes

- All models are saved in the `models/` directory.
- The default model path is `models/high_accuracy_model.walle`.
- Training uses the `tiny_stories_sample.json` dataset by default.
- By default, training uses 4000 stories from the dataset, but this can be customized with the `--stories` parameter.
- Memory optimization provides better performance on machines with limited memory bandwidth.
- When using larger batch sizes, consider increasing the number of curriculum examples to ensure proper level advancement.
- If you encounter thread deadlocks during training, increase the `--watchdog-timeout` and `--data-threads` values.
- For large datasets, always use `--auto-resize-vocab` to handle unexpected vocabulary growth. 
- When training on JSON-formatted datasets like TinyStories, be sure to include the `--json-format` parameter. 