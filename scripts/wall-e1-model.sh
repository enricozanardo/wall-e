#!/bin/bash

# Wall-E1 Model Management Script
# This script provides a simple interface for training, testing, and using Wall-E1 models.

set -e

# Display usage information
show_usage() {
  echo "Usage: $0 [command] [options]"
  echo ""
  echo "Commands:"
  echo "  train [--size small|medium|large]  Train a new model with specified size"
  echo "  generate [prompt] [options]        Generate text from a prompt"
  echo "  clean                              Remove all model files"
  echo ""
  echo "Generate options:"
  echo "  --model [path]                     Model file path (default: models/high_accuracy_model.walle)"
  echo "  --max-tokens [num]                 Maximum tokens to generate (default: 50)"
  echo ""
  echo "Examples:"
  echo "  $0 train --size small              Train a small model"
  echo "  $0 generate \"Once upon a time\"     Generate text from the default model"
  echo "  $0 generate \"Hello world\" --model models/my_model.walle --max-tokens 100"
}

# Ensure models directory exists
mkdir -p models

# Function to train a model
train_model() {
  local size="small"
  
  # Process arguments
  while [[ $# -gt 0 ]]; do
    case $1 in
      --size)
        size="$2"
        shift 2
        ;;
      *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
  done
  
  # Call the training script
  echo "Training a $size model..."
  ./scripts/train_optimized_accuracy.sh --size "$size"
  
  echo "Training complete! Model saved to models/high_accuracy_model.walle"
}

# Function to generate text
generate_text() {
  local prompt=""
  local model="models/high_accuracy_model.walle"
  local max_tokens=50
  
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
      *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
  done
  
  # Call the test generation script
  echo "Generating text from prompt: '$prompt'"
  ./scripts/test_generation.sh "$prompt" "$max_tokens" "$model"
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