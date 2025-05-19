# Wall-E1 Model Scripts

This directory contains scripts for training, managing, and generating text with Wall-E1 language models.

## Main Script

The `wall-e1-model.sh` script provides a unified interface for all Wall-E1 model operations:

```bash
# Train a new model
./wall-e1-model.sh train --size small|medium|large [--stories <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>] [--profile|--perf-log]

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

# Train with specific gradient checkpointing strategy
./wall-e1-model.sh train --size medium --memory-opt --checkpoint-strategy uniform

# Train with thread optimization for matrix multiplication
./wall-e1-model.sh train --size medium --thread-opt matrix_multiply

# Train with custom number of epochs
./wall-e1-model.sh train --size small --epochs 5

# Train with more curriculum examples to prevent stalling in level advancement
./wall-e1-model.sh train --size large --curriculum-examples 5000

# Train with all options
./wall-e1-model.sh train --size large --stories 5000 --cpus 8 --memory-opt --checkpoint-strategy adaptive --thread-opt gradient_update --batch-size 128 --epochs 15 --curriculum-examples 5000

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
- **--profile, --perf-log**: Enable detailed performance profiling and metrics collection

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

## Notes

- All models are saved in the `models/` directory.
- The default model path is `models/high_accuracy_model.walle`.
- Training uses the `tiny_stories_sample.json` dataset by default.
- By default, training uses 4000 stories from the dataset, but this can be customized with the `--stories` parameter.
- Memory optimization provides better performance on machines with limited memory bandwidth.
- When using larger batch sizes, consider increasing the number of curriculum examples to ensure proper level advancement. 