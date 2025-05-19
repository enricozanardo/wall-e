#!/bin/bash

# Wall-E1 Model Management Script
# This script provides a simple interface for training, testing, and using Wall-E1 models.

set -e

# Display usage information
show_usage() {
  echo "Usage: $0 [command] [options]"
  echo ""
  echo "Commands:"
  echo "  train [--size small|medium|large] [--stories <number>] [--cpus <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>] [--profile]  Train a new model with specified options"
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
  echo ""
  echo "Generate options:"
  echo "  --model [path]                     Model file path (default: models/high_accuracy_model.walle)"
  echo "  --max-tokens [num]                 Maximum tokens to generate (default: 50)"
  echo "  --cpus [number]                    Number of CPU cores to use (default: all available)"
  echo ""
  echo "Examples:"
  echo "  $0 train --size small              Train a small model with default stories"
  echo "  $0 train --size medium --stories 2000 --cpus 4 --memory-opt  Train a medium model with memory optimization"
  echo "  $0 train --size large --epochs 20  Train a large model with 20 epochs"
  echo "  $0 train --size medium --memory-opt --checkpoint-strategy uniform  Train with uniform checkpointing"
  echo "  $0 train --size large --thread-opt matrix_multiply  Optimize thread allocation for matrix multiplication"
  echo "  $0 train --size large --curriculum-examples 5000  Train with 5000 examples for curriculum"
  echo "  $0 train --size small --profile    Train a small model with performance profiling"
  echo "  $0 generate \"Once upon a time\"     Generate text from the default model"
  echo "  $0 generate \"Hello world\" --model models/my_model.walle --max-tokens 100 --cpus 2"
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
      --profile)
        profile="true"
        shift 1
        ;;
      --perf-log)
        profile="true"
        shift 1
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
  if [[ -n "$stories" ]]; then
    echo "Using $stories stories for training"
  fi
  if [[ -n "$cpus" ]]; then
    echo "Using $cpus CPU cores for training"
  fi
  if [[ -n "$memory_opt" ]]; then
    echo "Memory optimization enabled"
  fi
  if [[ -n "$checkpoint_strategy" ]]; then
    echo "Using $checkpoint_strategy gradient checkpointing strategy"
  fi
  if [[ -n "$thread_opt" ]]; then
    echo "Optimizing thread allocation for $thread_opt operations"
  fi
  if [[ -n "$batch_size" ]]; then
    echo "Using manual batch size: $batch_size"
  fi
  if [[ -n "$epochs" ]]; then
    echo "Using $epochs training epochs"
  fi
  if [[ -n "$curriculum_examples" ]]; then
    echo "Using $curriculum_examples examples for curriculum initialization"
  fi
  if [[ -n "$profile" ]]; then
    echo "Performance profiling enabled"
  fi
  
  # Check if data file exists
  local data_file="./data/tinystories.json"
  
  # First check for the 10k version
  if [[ -f "./data/tinystories-10k.json" ]]; then
    data_file="./data/tinystories-10k.json"
  elif [[ -f "./data/tinystories-1k.json" ]]; then
    data_file="./data/tinystories-1k.json"
  elif [[ ! -f "$data_file" ]]; then
    echo "Error: Training data file not found. Expected $data_file"
    echo "Please place the TinyStories JSON dataset in the data directory."
    exit 1
  fi

  # Set up save path
  local save_path="models/high_accuracy_model.walle"
  
  if [[ -n "$profile" ]]; then
    # Use our profiling script
    echo "Running with performance profiling..."
    
    # Create profiling directory if it doesn't exist
    mkdir -p ./profiling_results
    
    ./scripts/profile_training.sh \
      $stories_param \
      --model-dim $model_dim \
      --ff-dim $ff_dim \
      --layers $layers \
      $epochs_param \
      $cpus_param \
      $memory_opt_param \
      $checkpoint_strategy_param \
      $thread_opt_param \
      $batch_size_param \
      $curriculum_examples_param \
      $data_file
  else
    # Use the regular training script
    ./scripts/train_optimized_accuracy.sh \
      --size "$size" \
      $stories_param \
      $cpus_param \
      $memory_opt_param \
      $checkpoint_strategy_param \
      $thread_opt_param \
      $batch_size_param \
      $epochs_param \
      $curriculum_examples_param
  fi
  
  echo "Training complete! Model saved to models/high_accuracy_model.walle"
}

# Function to generate text
generate_text() {
  local prompt=""
  local model="models/high_accuracy_model.walle"
  local max_tokens=50
  local cpus=""
  local cpus_param=""
  
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
      *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
  done
  
  # Call the test generation script
  echo "Generating text from prompt: '$prompt'"
  ./scripts/test_generation.sh "$prompt" "$max_tokens" "$model" $cpus_param
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