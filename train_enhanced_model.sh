#!/bin/bash
# Enhanced training script with improved anti-repetition mechanisms and curriculum learning
# Based on the new EnhancedTrainer implementation

echo "===== WALL-E1 ENHANCED TRAINING SCRIPT ====="
echo "This script uses the EnhancedTrainer with improved tokenization and repetition prevention"

# Create models directory if it doesn't exist
mkdir -p models

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Default parameters
MODEL_DIM=192
FF_DIM=768
HEADS=6
LAYERS=4
EPOCHS=10
VOCAB_SIZE=5000
SAVE_PATH="models/enhanced_model.json"
ENABLE_SKIP=true
STRONG_ANTI_REP=true
MIN_FREQ=2
JSON_FORMAT=true
MAX_STORIES=8000  # Process up to this many stories from the JSON dataset

# Check for quick mode flag
if [ "$1" == "--quick" ]; then
  echo "Using quick training mode with smaller model"
  MODEL_DIM=128
  FF_DIM=512
  HEADS=4
  LAYERS=3
  EPOCHS=5
  MAX_STORIES=4000
  shift
fi

# Default to TinyStories data
DATA_PATH="${1:-data/tiny_stories_sample.json}"
if [ ! -f "$DATA_PATH" ]; then
  # Try alternate data files
  ALTERNATES=("data/tiny_stories_sample_updated.json" "data/story.txt" "data/qa_en_dataset.json")
  
  for alt in "${ALTERNATES[@]}"; do
    if [ -f "$alt" ]; then
      echo "Using alternate data file: $alt"
      DATA_PATH="$alt"
      break
    fi
  done
  
  # If still no file found
  if [ ! -f "$DATA_PATH" ]; then
    echo "WARNING: No suitable training data file found! Please provide a valid path."
    exit 1
  fi
fi

# Determine if we're using JSON format based on file extension
if [[ "$DATA_PATH" == *.json ]]; then
  JSON_FORMAT=true
  echo "Detected JSON format data file"
else
  JSON_FORMAT=false
  echo "Detected plain text data file"
fi

echo "Starting enhanced training with improved tokenization and anti-repetition..."
echo "Training data: $DATA_PATH"
echo "Model parameters: dim=$MODEL_DIM, ff=$FF_DIM, heads=$HEADS, layers=$LAYERS"

# Build command
CMD="cargo run --release --bin train_enhanced_model -- $DATA_PATH"

# Add model parameters
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --save-path $SAVE_PATH"
CMD="$CMD --min-freq $MIN_FREQ"

# Add additional parameters for JSON format
if [ "$JSON_FORMAT" = true ]; then
  CMD="$CMD --json-format --stories $MAX_STORIES"
fi

# Add skip connections if enabled
if [ "$ENABLE_SKIP" = true ]; then
  CMD="$CMD --enable-skip"
fi

# Add strong anti-repetition if enabled
if [ "$STRONG_ANTI_REP" = true ]; then
  CMD="$CMD --strong-anti-rep"
fi

# Print pretty header
echo "╔═════════════════════════════════════════════════════╗"
echo "║            WALL-E1 ENHANCED TRAINING                ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers"
echo "Skip connections: $ENABLE_SKIP"
echo "Enhanced anti-repetition: $STRONG_ANTI_REP"
echo "Training for $EPOCHS epochs"
echo "Data format: $(if [ "$JSON_FORMAT" = true ]; then echo "JSON (max $MAX_STORIES stories)"; else echo "Plain text"; fi)"
echo "╔═════════════════════════════════════════════════════╗"
echo "║               Executing command                     ║"
echo "╚═════════════════════════════════════════════════════╝"

# Run the command
echo "$CMD"
$CMD

# Check result
if [ $? -eq 0 ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║              TRAINING COMPLETE                      ║"
  echo "║                                                     ║"
  echo "║  Model saved to: $SAVE_PATH"
  echo "╚═════════════════════════════════════════════════════╝"
  echo "You can test the model with:"
  echo "cargo run --release --bin generate -- --model $SAVE_PATH --prompt \"Once upon a time\""
else
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║                TRAINING FAILED                      ║"
  echo "╚═════════════════════════════════════════════════════╝"
  echo "Check the error messages above for details."
fi 