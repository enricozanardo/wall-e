# Wall-E1 Model Scripts

This directory contains scripts for training, managing, and generating text with Wall-E1 language models.

## Main Script

The `wall-e1-model.sh` script provides a unified interface for all Wall-E1 model operations:

```bash
# Train a new model
./wall-e1-model.sh train --size small|medium|large [--stories <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>] [--auto-resize-vocab] [--watchdog-timeout <seconds>] [--batch-timeout <seconds>] [--data-threads <number>] [--target-id-max <number>] [--vocab-size <number>] [--min-freq <number>] [--enable-skip] [--strong-anti-rep] [--json-format] [--parallel] [--profile|--perf-log]

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

# Train with parallel data preparation for better performance
./wall-e1-model.sh train --size medium --parallel

# Train with all options
./wall-e1-model.sh train --size large --stories 5000 --cpus 8 --memory-opt --checkpoint-strategy adaptive --thread-opt gradient_update --batch-size 128 --epochs 15 --curriculum-examples 5000 --auto-resize-vocab --target-id-max 10000 --watchdog-timeout 180 --batch-timeout 60 --data-threads 12 --vocab-size 5000 --min-freq 2 --enable-skip --strong-anti-rep --json-format --parallel

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
- **--batch-timeout <seconds>**: Set timeout for individual batch processing in multi-threaded mode (default: 60). Helps prevent threads from hanging in computation.
- **--data-threads <number>**: Number of threads for data loading (min: 8, default: 70% of available cores). Increasing can help with CPU utilization.
- **--target-id-max <number>**: Maximum target ID value (default: auto-detected, min: 5000). Set higher for larger vocabularies.
- **--vocab-size <number>**: Size of the vocabulary (default: 10000). Smaller values create a more compact model, larger values improve accuracy but increase memory usage.
- **--min-freq <number>**: Minimum token frequency for vocabulary inclusion (default: 2). Higher values create a more focused vocabulary but may increase out-of-vocabulary tokens.
- **--enable-skip**: Enable skip connections in the model architecture, which can improve gradient flow and model performance.
- **--strong-anti-rep**: Enable stronger anti-repetition mechanisms to prevent the model from generating repetitive text.
- **--json-format**: Use JSON format for input data instead of plain text. Required when training on TinyStories or similar JSON-formatted datasets.
- **--parallel**: Enable parallel data preparation while maintaining reliable training. This accelerates data processing but keeps the model updates safe and stable.
- **--profile, --perf-log [true|false]**: Enable detailed performance profiling and metrics collection. You can use it as a flag (--perf-log) or with an explicit value (--perf-log true)

## Reliable Training

Wall-E1 now uses a reliable training mechanism by default to avoid thread synchronization issues that can lead to deadlocks and stalled training:

1. **Single-Threaded Training**: The system now uses a fully sequential training process that eliminates thread synchronization issues by design. This approach provides:
   - Much higher reliability by avoiding deadlocks entirely
   - Consistent, predictable training behavior
   - Simpler debugging when issues occur

2. **Parallel Data Preparation**: The `--parallel` option allows safe parallelization of data preparation while maintaining reliable single-threaded model training. This gives you:
   - The stability of single-threaded model updates
   - Performance benefits of parallel data preparation
   - Balanced approach for most training scenarios

To enable parallel data preparation with reliable training:

```bash
./wall-e1-model.sh train --size medium --parallel
```

The parallel option is especially recommended for larger datasets or when performance is important, as it provides a significant speedup while maintaining the reliability of the training process.

## Training Modes

Wall-E1 offers three distinct training modes with different performance and reliability characteristics:

1. **Regular Single-Threaded Training (Default)**: 
   - Uses a single thread for all training operations
   - Highest reliability and consistency
   - Predictable memory usage
   - Slowest performance, especially on multi-core systems
   - Usage: `./wall-e1-model.sh train --size medium`

2. **Parallel Data Preparation Mode**:
   - Uses multiple threads for data loading, tokenization, and batch preparation
   - Single-threaded model update for reliability
   - Good balance of performance and stability
   - Recommended for most use cases
   - Usage: `./wall-e1-model.sh train --size medium --parallel`

3. **Multi-Threaded Training Mode (Experimental)**:
   - Full multi-threading for both data preparation and model training
   - Synchronized gradient updates using thread barriers
   - Potential for significant performance improvement on multi-core systems
   - May encounter stability issues on some hardware configurations
   - Usage: `./wall-e1-model.sh train --size medium --mt-training`

Each mode has specific use cases:
- Use the default mode when stability is critical and performance is less important
- Use parallel data preparation mode for a good balance of performance and reliability
- Use multi-threaded mode when maximum performance is required and you can tolerate some potential instability

The system will automatically allocate appropriate thread pools for each mode based on your CPU architecture.

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
5. **Reliable Training Mode**: Now used by default, providing a fully sequential training approach that eliminates thread synchronization issues by design.

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
- Reliable training is now used by default, which has solved the thread deadlock issues that occurred in previous versions.
- For large datasets, always use `--auto-resize-vocab` to handle unexpected vocabulary growth. 
- When training on JSON-formatted datasets like TinyStories, be sure to include the `--json-format` parameter.

## Wall-E1 Training Scripts

This directory contains scripts for training and evaluating Wall-E1 models.

### Main Command Script

The `wall-e1-model.sh` script is the main entry point for training and evaluating models. Use it as follows:

```bash
./wall-e1-model.sh [command] [options]
```

Commands:
- `train`: Train a new model
- `profile`: Train with performance metrics collection
- `clean`: Remove temporary files

### Training Options

The `train` command supports the following options:

```bash
./wall-e1-model.sh train [--size tiny|small|medium|large] [--model-dim N] [--ff-dim N] 
                        [--heads N] [--layers N] [--dropout N] [--epochs N] [--vocab-size N] 
                        [--min-freq N] [--batch-size N] [--save-path FILE] [--learning-rate N]
                        [--stories N] [--cpus N] [--memory-opt] [--disable-curriculum] 
                        [--enable-skip] [--strong-anti-rep] [--json-format] [--parallel] [--mt-training]
```

### Common Examples

Train a small model with default settings:
```bash
./wall-e1-model.sh train --size small
```

Train a medium model with parallel data preparation:
```bash
./wall-e1-model.sh train --size medium --parallel
```

Train with multi-threaded model training (experimental):
```bash
./wall-e1-model.sh train --size small --mt-training
```

Custom configuration:
```bash
./wall-e1-model.sh train --model-dim 256 --ff-dim 1024 --layers 4 --epochs 10 --stories 500 --memory-opt --enable-skip --strong-anti-rep --json-format --parallel
```

### Profiling Options

The `profile` command accepts the same options as `train` but will collect detailed performance metrics:

```bash
./wall-e1-model.sh profile --size tiny --stories 50 --parallel
```

This will run a short training session with performance metrics collection, saving results to the `profiling_results` directory.

## Multi-Threaded Training (Experimental)

For maximum performance on multi-core systems, Wall-E1 now includes a full multi-threaded training implementation with the following improvements:

1. **Efficient Thread Utilization**: The system automatically determines the optimal number of worker threads based on batch size and available CPU cores to maximize parallelism without excessive thread overhead.

2. **Enhanced Gradient Computation**: The `train_step_compute_only` method uses parallel iterators and column-wise normalization for better CPU utilization during gradient computation.

3. **Improved Synchronization**: Replaced traditional barriers with atomic counters and mutex-based synchronization to prevent deadlocks and ensure reliable operation.

4. **Thread State Tracking**: Detailed thread state tracking and reporting helps identify bottlenecks and potential issues during training.

5. **Automatic Timeout Detection**: Built-in watchdog monitors ensure that training doesn't stall indefinitely due to thread synchronization issues.

To enable multi-threaded training, use the `--mt-training` or `--mt` flag:

```bash
./wall-e1-model.sh train --size small --mt-training
```

This mode will parallelize both the data preparation and the gradient computation/model updates, resulting in improved training speeds on multi-core systems.

**NOTE:** Multi-threaded training is still experimental and may not be stable for all model configurations. Best results are achieved with larger batch sizes that allow for better workload distribution across threads.

## Performance Optimization Features

Wall-E1 offers three distinct training modes to balance reliability and performance:

1. **Reliable Training (Default)**: The standard training approach, which ensures consistent results by using a single thread for model updates.
   - Uses sequential processing for both data preparation and model updates
   - Completely eliminates thread synchronization issues by design
   - Best option for debugging or when absolute reliability is required
   - Slowest performance, especially on multi-core systems

2. **Parallel Data Preparation**: The `--parallel` option allows safe parallelization of data preparation while maintaining reliable single-threaded model training. This gives you:
   - Parallel data loading, tokenization, and batch creation
   - Single-threaded model parameter updates for reliability
   - Up to 2-3x speedup over default mode, depending on your hardware
   - Good balance of performance and reliability for most use cases

3. **Multi-Threaded Model Training**: The new `--mt-training` option enables full multi-threaded model training:
   - Parallel batch processing across multiple CPU cores
   - Thread-safe gradient accumulation with atomic operations
   - Synchronized weight updates using coordination barriers
   - Watchdog monitoring to detect potential deadlocks
   - Up to 4-5x speedup over default mode on many-core systems
   - Currently experimental - may not be stable in all configurations

### Usage Recommendations

For maximum reliability (e.g., when debugging):
```bash
./wall-e1-model.sh train --size medium
```

For good performance while maintaining reliability (recommended for most use cases):
```bash
./wall-e1-model.sh train --size medium --parallel
```

For maximum performance (experimental):
```bash
./wall-e1-model.sh train --size medium --mt-training
```

The optimal choice depends on your hardware, dataset size, and requirements:
- Single-core or dual-core systems may see limited benefit from `--mt-training`
- Systems with 4+ cores will see significant improvement with `--parallel`
- Systems with 8+ cores will benefit most from `--mt-training` for large models
- Always use `--parallel` at minimum when training medium or large models

## Implementation Details

The scripts in this directory wrap the Rust-based Wall-E1 training system, providing convenient access to the most common training options. The implementation includes:

1. `wall-e1-model.sh` - Main entry point script
2. `train_optimized_accuracy.sh` - Focused on reliable training
3. `profile_training.sh` - Performance profiling script

These scripts automatically configure optimal thread pool sizes, memory allocation strategies, and other low-level optimizations based on your system and the selected options. 

## Known Issues

### Multi-threaded Training Implementation

The test program confirms that multi-threaded training is properly implemented and works correctly when using the test binary directly:

```bash
# Run the multi-threading test to verify it's working:
cargo run --release --bin test_multithreading
```

The test will validate that:
1. Multi-threaded training provides a significant speedup (5-6x) over single-threaded training
2. All CPU cores are utilized during training
3. The batch timeout mechanism effectively prevents deadlocks

However, there's currently an issue with the script-based approach:

1. There's a mismatch between script parameters and binary parameters. When running through `./scripts/wall-e1-model.sh`, the training command will fail with an "Unknown option: --dataset" error.
2. The parameters aren't being properly passed from one script to another.

**Workaround:**
If you need to use multi-threaded training, use the test_multithreading binary directly for now, or modify your own version of the scripts to fix the parameter handling issues. 