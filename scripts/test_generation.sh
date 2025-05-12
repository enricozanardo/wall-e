#!/bin/bash

# Test script for text generation with the Wall-E1 model

# Set variables
PROMPT=${1:-"In a world where dragons"}
MAX_TOKENS=${2:-50}
MODEL=${3:-"models/high_accuracy_model.walle"}
OUTPUT_FILE="generated_text.txt"
LOG_FILE="generation_log.txt"

echo "Running text generation with prompt: '$PROMPT'"
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
    --max-tokens $MAX_TOKENS 2>&1 | tee "$LOG_FILE"

echo "================================================================"
echo "Contents of $OUTPUT_FILE:"
if [ -f "$OUTPUT_FILE" ]; then
    cat "$OUTPUT_FILE"
else
    echo "Output file not found!"
fi
echo "================================================================"
echo "Test completed, see $LOG_FILE for full log" 