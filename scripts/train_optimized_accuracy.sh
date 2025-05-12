#!/bin/bash

# Script for optimized training with accuracy focus

# Default model size
MODEL_SIZE="small"
# Output log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
# Ensure log directory exists
mkdir -p logs

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--size small|medium|large]"
      exit 1
      ;;
  esac
done

# Set model parameters based on size
case $MODEL_SIZE in
  "small")
    MODEL_DIM=128
    FF_DIM=512
    HEADS=4
    LAYERS=3
    ;;
  "medium")
    MODEL_DIM=256
    FF_DIM=1024
    HEADS=8
    LAYERS=6
    ;;
  "large")
    MODEL_DIM=512
    FF_DIM=2048
    HEADS=8
    LAYERS=8
    ;;
  *)
    echo "Invalid model size: $MODEL_SIZE"
    echo "Valid options: small, medium, large"
    exit 1
    ;;
esac

# Display fancy header
echo "Detected JSON format data file" | tee -a "$LOG_FILE"
echo "Using quality-focused sampling: 4000 high-quality stories" | tee -a "$LOG_FILE"

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║              STEP 1: DATA PREPARATION               ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

echo "Preprocessing disabled. Using original data." | tee -a "$LOG_FILE"

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║          OPTIMIZED TRAINING CONFIGURATION           ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

echo "Model size: $MODEL_SIZE" | tee -a "$LOG_FILE"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers" | tee -a "$LOG_FILE"
echo "Feed-forward dimension: $FF_DIM" | tee -a "$LOG_FILE"
echo "Vocabulary size: 5000" | tee -a "$LOG_FILE"
echo "Min token frequency: 2" | tee -a "$LOG_FILE"
echo "Learning rate: 0.0001" | tee -a "$LOG_FILE"
echo "Epochs: 10" | tee -a "$LOG_FILE"
echo "Data sampling: quality (4000 stories)" | tee -a "$LOG_FILE"
echo "Data preprocessing: false" | tee -a "$LOG_FILE"
echo "Training data: data/tiny_stories_sample.json" | tee -a "$LOG_FILE"
echo "Save path: models/high_accuracy_model.json" | tee -a "$LOG_FILE"

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║           BUILDING OPTIMIZED COMMAND                ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║           STARTING OPTIMIZED TRAINING               ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

# Execute the optimized training command
TRAINING_CMD="cargo run --release --bin train_enhanced_model -- data/tiny_stories_sample.json --model-dim $MODEL_DIM --ff-dim $FF_DIM --heads $HEADS --layers $LAYERS --epochs 10 --vocab-size 5000 --min-freq 2 --save-path models/high_accuracy_model.json --learning-rate 0.0001 --enable-skip --strong-anti-rep --json-format --stories 4000"

echo "Executing: $TRAINING_CMD" | tee -a "$LOG_FILE"
eval $TRAINING_CMD | tee -a "$LOG_FILE"

# Function to generate text with the trained model
generate_text() {
  local prompt="$1"
  echo "Prompt: \"$prompt\"" | tee -a "$LOG_FILE"
  
  # Note the ordering of arguments - make sure --model comes before the path
  # and generate-only is a flag without a value
  GENERATE_CMD="cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.json --prompt \"$prompt\" --max-tokens 75"
  eval $GENERATE_CMD | tee -a "$LOG_FILE"
}

echo "Generating sample text to demonstrate model quality..." | tee -a "$LOG_FILE"
generate_text "Once upon a time"
generate_text "The quick brown fox"
generate_text "In a world where magic"
generate_text "The most important thing"

echo "Training complete. You can test the model with:" | tee -a "$LOG_FILE"
echo "cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.json --prompt \"Your prompt here\" --max-tokens 100" | tee -a "$LOG_FILE" 