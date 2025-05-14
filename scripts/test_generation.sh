#!/bin/bash

# Test script for text generation with the Wall-E1 model

# Set variables
PROMPT=${1:-"In a world where dragons"}
MAX_TOKENS=${2:-50}
MODEL=${3:-"models/high_accuracy_model.walle"}
CPU_PARAM=""

# Process any additional parameters
shift 3 2>/dev/null || true
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpus)
      if [ "$2" -gt 0 ] 2>/dev/null; then
        export RAYON_NUM_THREADS=$2
        echo "Using $2 CPU cores for text generation"
        CPU_PARAM="--perf-log true"
      fi
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# If CPU cores not explicitly set, use all available
if [ -z "$RAYON_NUM_THREADS" ]; then
  AVAILABLE_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  export RAYON_NUM_THREADS=$AVAILABLE_CPUS
  echo "Using all available CPU cores ($AVAILABLE_CPUS) for text generation"
  CPU_PARAM="--perf-log true"
fi

OUTPUT_FILE="generated_text.txt"
LOG_FILE="generation_log.txt"

echo "Running text generation with prompt: '$PROMPT'"
echo "Using RAYON_NUM_THREADS=$RAYON_NUM_THREADS"
echo "================================================================"

# Ensure models directory exists
mkdir -p models

# If the test model doesn't exist, suggest running the training script
if [ ! -f "$MODEL" ]; then
    echo "Model file not found: $MODEL"
    echo "Please run the training script first:"
    echo "./scripts/train_optimized_accuracy.sh --size small"
    exit 1
fi

# Build and run the model in release mode for better performance
cargo build --release

# Run the text generation with logging
cargo run --release --bin train_enhanced_model -- \
    --generate-only \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --max-tokens $MAX_TOKENS \
    $CPU_PARAM 2>&1 | tee "$LOG_FILE"

echo "================================================================"
echo "Contents of $OUTPUT_FILE:"
if [ -f "$OUTPUT_FILE" ]; then
    cat "$OUTPUT_FILE"
else
    echo "Output file not found!"
fi
echo "================================================================"
echo "Test completed, see $LOG_FILE for full log" 