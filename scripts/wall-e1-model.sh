#!/bin/bash

# Wall-E1 Model Management Script
# This script provides a simple interface for training, testing, and using Wall-E1 models.

set -e

# Display usage information
show_usage() {
  echo "Usage: $0 [command] [options]"
  echo ""
  echo "Commands:"
  echo "  train [--size small|medium|large] [--stories <number>] [--cpus <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>] [--profile] [--auto-resize-vocab] [--watchdog-timeout <seconds>] [--batch-timeout <seconds>] [--data-threads <number>] [--target-id-max <number>] [--vocab-size <number>] [--min-freq <number>] [--enable-skip] [--strong-anti-rep] [--json-format] [--parallel] [--mt-training] Train a new model with specified options"
  echo "  generate [prompt] [options]        Generate text from a prompt"
  echo "  clean                              Remove all model files"
  echo ""
  echo "Train options:"
  echo "  --size [size]                      Model size: small, medium or large (default: small)"
  echo "  --stories [number]                 Number of stories to use for training (default: 4000)"
  echo "  --cpus [number]                    Number of CPU cores to use (default: all available)"
  echo "  --memory-opt                       Enable memory optimization (optimal batch size, thread allocation)"
  echo "  --checkpoint-strategy [strategy]   Gradient checkpointing strategy: boundary, uniform, adaptive (default: adaptive)"
  echo "  --thread-opt [operation]           Optimize thread allocation for specific operation: matrix_multiply, attention, gradient_update, data_loading"
  echo "  --batch-size [number]              Manually set batch size (overrides automatic calculation)"
  echo "  --epochs [number]                  Number of training epochs (default: 10)"
  echo "  --curriculum-examples [number]     Number of examples to use for curriculum initialization (default: 500)"
  echo "  --profile, --perf-log              Enable detailed performance profiling"
  echo "  --auto-resize-vocab                Automatically resize vocabulary for out-of-range target IDs"
  echo "  --watchdog-timeout [seconds]       Set timeout for watchdog thread detection (default: 60)"
  echo "  --batch-timeout [seconds]         Set timeout for individual batch processing in multi-threaded mode (default: 60)"
  echo "  --data-threads [number]            Number of threads for data loading (min: 8, default: 70% of available cores)"
  echo "  --target-id-max [number]           Maximum target ID value (default: auto-detected, min: 5000)"
  echo "  --vocab-size [number]              Size of the vocabulary (default: 10000)"
  echo "  --min-freq [number]                Minimum token frequency for vocabulary inclusion (default: 2)"
  echo "  --enable-skip                      Enable skip connections in the model architecture"
  echo "  --strong-anti-rep                  Enable stronger anti-repetition mechanisms"
  echo "  --json-format                      Use JSON format for input data instead of plain text"
  echo "  --parallel                         Enable parallel data preparation (for faster training)"
  echo "  --mt-training                      Enable multi-threaded model training (experimental)"
  echo ""
  echo "Generate options:"
  echo "  --model [path]                     Model file path (default: models/high_accuracy_model.walle)"
  echo "  --max-tokens [num]                 Maximum tokens to generate (default: 50)"
  echo "  --cpus [number]                    Number of CPU cores to use (default: all available)"
  echo "  --disable-watchdog                 Disable watchdog during text generation to prevent stalled progress warnings"
  echo ""
  echo "Examples:"
  echo "  $0 train --size small              Train a small model with default stories"
  echo "  $0 train --size medium --stories 2000 --cpus 4 --memory-opt  Train a medium model with memory optimization"
  echo "  $0 train --size large --epochs 20  Train a large model with 20 epochs"
  echo "  $0 train --size medium --memory-opt --checkpoint-strategy uniform  Train with uniform checkpointing"
  echo "  $0 train --size large --thread-opt matrix_multiply  Optimize thread allocation for matrix multiplication"
  echo "  $0 train --size large --curriculum-examples 5000  Train with 5000 examples for curriculum"
  echo "  $0 train --size small --profile    Train a small model with performance profiling"
  echo "  $0 train --size medium --auto-resize-vocab --target-id-max 10000  Train with automatic vocabulary resizing"
  echo "  $0 train --size small --watchdog-timeout 120 --data-threads 16  Train with custom watchdog and data thread settings"
  echo "  $0 train --size small --vocab-size 5000 --min-freq 2 --enable-skip --strong-anti-rep --json-format  Train with custom vocabulary and architecture settings"
  echo "  $0 train --size small --parallel   Train with parallel data preparation for better performance"
  echo "  $0 train --size small --mt-training --batch-timeout 90  Train with multi-threading and custom batch timeout"
  echo "  $0 generate \"Once upon a time\"     Generate text from the default model"
  echo "  $0 generate \"Hello world\" --model models/my_model.walle --max-tokens 100 --cpus 2 --disable-watchdog"
}

# Ensure models directory exists
mkdir -p models

# Function to train a model
train_model() {
  local size="small"
  local stories=""
  local stories_param=""
  local cpus=""
  local cpus_param=""
  local memory_opt=""
  local memory_opt_param=""
  local checkpoint_strategy=""
  local checkpoint_strategy_param=""
  local thread_opt=""
  local thread_opt_param=""
  local batch_size=""
  local batch_size_param=""
  local epochs=""
  local epochs_param=""
  local curriculum_examples=""
  local curriculum_examples_param=""
  local profile=""
  local auto_resize_vocab=""
  local auto_resize_vocab_param=""
  local watchdog_timeout=""
  local watchdog_timeout_param=""
  local batch_timeout=""
  local batch_timeout_param=""
  local data_threads=""
  local data_threads_param=""
  local target_id_max=""
  local target_id_max_param=""
  # New parameters
  local vocab_size=""
  local vocab_size_param=""
  local min_freq=""
  local min_freq_param=""
  local enable_skip=""
  local enable_skip_param=""
  local strong_anti_rep=""
  local strong_anti_rep_param=""
  local json_format=""
  local json_format_param=""
  # New reliable training parameters
  local reliable_training=""
  local reliable_training_param=""
  local parallel_data_prep=""
  local parallel_data_prep_param=""
  local mt_training_param=""
  
  # Process arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --size)
        size="$2"
        shift 2
        ;;
      --stories)
        stories="$2"
        stories_param="--stories $stories"
        shift 2
        ;;
      --cpus)
        cpus="$2"
        cpus_param="--cpus $cpus"
        shift 2
        ;;
      --memory-opt)
        memory_opt="true"
        memory_opt_param="--memory-opt"
        shift 1
        ;;
      --checkpoint-strategy)
        checkpoint_strategy="$2"
        checkpoint_strategy_param="--checkpoint-strategy $checkpoint_strategy"
        shift 2
        ;;
      --thread-opt)
        thread_opt="$2"
        thread_opt_param="--thread-opt $thread_opt"
        shift 2
        ;;
      --batch-size)
        batch_size="$2"
        batch_size_param="--batch-size $batch_size"
        shift 2
        ;;
      --epochs)
        epochs="$2"
        epochs_param="--epochs $epochs"
        shift 2
        ;;
      --curriculum-examples)
        curriculum_examples="$2"
        curriculum_examples_param="--curriculum-examples $curriculum_examples"
        shift 2
        ;;
      --auto-resize-vocab)
        auto_resize_vocab="true"
        auto_resize_vocab_param="--auto-resize-vocab"
        shift 1
        ;;
      --watchdog-timeout)
        watchdog_timeout="$2"
        watchdog_timeout_param="--watchdog-timeout $watchdog_timeout"
        shift 2
        ;;
      --batch-timeout)
        batch_timeout="$2"
        batch_timeout_param="--batch-timeout $batch_timeout"
        shift 2
        ;;
      --data-threads)
        data_threads="$2"
        data_threads_param="--data-threads $data_threads"
        shift 2
        ;;
      --target-id-max)
        target_id_max="$2"
        target_id_max_param="--target-id-max $target_id_max"
        shift 2
        ;;
      --vocab-size)
        vocab_size="$2"
        vocab_size_param="--vocab-size $vocab_size"
        shift 2
        ;;
      --min-freq)
        min_freq="$2"
        min_freq_param="--min-freq $min_freq"
        shift 2
        ;;
      --enable-skip)
        enable_skip="true"
        enable_skip_param="--enable-skip"
        shift 1
        ;;
      --strong-anti-rep)
        strong_anti_rep="true"
        strong_anti_rep_param="--strong-anti-rep"
        shift 1
        ;;
      --json-format)
        json_format="true"
        json_format_param="--json-format"
        shift 1
        ;;
      --reliable-training)
        # Reliable training is now the default approach, so we'll just show a message
        echo "Note: Reliable training is now the default mode, ignoring redundant option"
        shift 1
        ;;
      --parallel-data-prep|--parallel)
        parallel_data_prep="true"
        parallel_data_prep_param="--parallel"
        shift 1
        ;;
      --profile)
        profile="true"
        shift 1
        ;;
      --perf-log)
        profile="true"
        # Check if the next argument is a value for perf-log
        if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
          # Accept any value, but only treat "true" as true
          if [[ "$2" == "true" ]]; then
            profile="true"
          elif [[ "$2" == "false" ]]; then
            profile="false"
          fi
          shift 2
        else
          # No value provided, treat as flag
          shift 1
        fi
        ;;
      --mt-training|--mt)
        echo "Enabling multi-threaded training (experimental)"
        mt_training_param="--mt-training"
        # Also make sure we're using parallel data preparation with MT training
        parallel_data_prep_param="--parallel"
        shift
        ;;
      *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
  done

  # Set model dimensions based on size
  local model_dim=""
  local ff_dim=""
  local layers=""
  
  case $size in
    small)
      model_dim="128"
      ff_dim="512"
      layers="2"
      ;;
    medium)
      model_dim="256"
      ff_dim="1024"
      layers="4"
      ;;
    large)
      model_dim="512"
      ff_dim="2048"
      layers="6"
      ;;
    *)
      echo "Unknown model size: $size"
      exit 1
      ;;
  esac
  
  # Call the training script
  echo "Training a $size model..."
  echo "Using $stories stories for training"
  echo "Using $epochs training epochs"
  
  # Output batch timeout information if set
  if [[ -n "$batch_timeout_param" ]]; then
    batch_timeout_val=${batch_timeout_param#--batch-timeout }
    echo "Batch timeout set to $batch_timeout_val seconds"
    export WALL_E_BATCH_TIMEOUT=$batch_timeout_val
  fi
  
  # Enable multi-threaded training explicitly if set
  if [[ "$mt_training_param" == "--mt-training" ]]; then
    echo "Multi-threaded training enabled"
    export WALL_E_MT_TRAINING=1
  fi
  
  # Run the training script with parameters
  ./scripts/train_optimized_accuracy.sh \
    --size $size \
    $stories_param \
    $cpus_param \
    $memory_opt_param \
    $checkpoint_strategy_param \
    $thread_opt_param \
    $batch_size_param \
    $epochs_param \
    $curriculum_examples_param \
    $auto_resize_vocab_param \
    $watchdog_timeout_param \
    $batch_timeout_param \
    $data_threads_param \
    $target_id_max_param \
    $vocab_size_param \
    $min_freq_param \
    $enable_skip_param \
    $strong_anti_rep_param \
    $json_format_param \
    $parallel_data_prep_param \
    $mt_training_param
  
  echo "Training complete! Model saved to models/high_accuracy_model.walle"
}

# Function to generate text
generate_text() {
  local prompt=""
  local model="models/high_accuracy_model.walle"
  local max_tokens=50
  local cpus=""
  local cpus_param=""
  local disable_watchdog=""
  local disable_watchdog_param=""
  
  # Get the prompt
  if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    prompt="$1"
    shift
  else
    echo "Error: No prompt provided"
    show_usage
    exit 1
  fi
  
  # Process remaining arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --model)
        model="$2"
        shift 2
        ;;
      --max-tokens)
        max_tokens="$2"
        shift 2
        ;;
      --cpus)
        cpus="$2"
        cpus_param="--cpus $cpus"
        shift 2
        ;;
      --disable-watchdog)
        disable_watchdog="true"
        disable_watchdog_param="--disable-watchdog"
        shift 1
        ;;
      *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
  done
  
  # Call the test generation script
  echo "Generating text from prompt: '$prompt'"
  ./scripts/test_generation.sh "$prompt" "$max_tokens" "$model" $cpus_param $disable_watchdog_param
}

# Function to clean models directory
clean_models() {
  echo "Removing all model files..."
  rm -f models/*.walle models/*.json models/*.bin models/*.epoch*
  echo "All model files have been removed."
}

# Main command processing
if [[ $# -lt 1 ]]; then
  show_usage
  exit 1
fi

command="$1"
shift

case "$command" in
  train)
    train_model "$@"
    ;;
  generate)
    generate_text "$@"
    ;;
  clean)
    clean_models
    ;;
  help|--help|-h)
    show_usage
    ;;
  *)
    echo "Unknown command: $command"
    show_usage
    exit 1
    ;;
esac 