#!/bin/bash
# Improved accuracy training script with more conservative parameters
# Based on analysis of previous training results

echo "===== WALL-E1 IMPROVED ACCURACY TRAINING SCRIPT ====="
echo "This script uses conservative parameters optimized for accuracy rather than capacity"

# Create models directory if it doesn't exist
mkdir -p models

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Default parameters - Using a smaller model for better accuracy
MODEL_DIM=96       # Reduced from 192 for better accuracy
FF_DIM=384         # Reduced from 768
HEADS=3            # Reduced from 6 
LAYERS=3           # Reduced from 4
EPOCHS=15          # Increased from 10 for better learning
VOCAB_SIZE=4000    # Reduced from 5000
SAVE_PATH="models/accurate_model.json"
ENABLE_SKIP=false  # Disable skip connections as they might interfere with learning
STRONG_ANTI_REP=false # Use standard anti-repetition to avoid overpenalizing
MIN_FREQ=3         # Increased to focus on more common tokens
JSON_FORMAT=true
MAX_STORIES=8000   # Process up to this many stories from the JSON dataset

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

echo "Starting training with conservative parameters optimized for accuracy..."
echo "Training data: $DATA_PATH"
echo "Model parameters: dim=$MODEL_DIM, ff=$FF_DIM, heads=$HEADS, layers=$LAYERS"

# Build command with a focus on accuracy
CMD="cargo run --release --bin train_enhanced_model -- $DATA_PATH"

# Add model parameters - smaller model for better accuracy
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --save-path $SAVE_PATH"
CMD="$CMD --min-freq $MIN_FREQ"
CMD="$CMD --learning-rate 0.0003" # Slightly lower learning rate for stability

# Add additional parameters for JSON format
if [ "$JSON_FORMAT" = true ]; then
  CMD="$CMD --json-format --stories $MAX_STORIES"
fi

# Add skip connections if enabled (disabled by default for accuracy)
if [ "$ENABLE_SKIP" = true ]; then
  CMD="$CMD --enable-skip"
fi

# Add strong anti-repetition if enabled (disabled by default for accuracy)
if [ "$STRONG_ANTI_REP" = true ]; then
  CMD="$CMD --strong-anti-rep"
fi

# Print header with information
echo "╔═════════════════════════════════════════════════════╗"
echo "║        WALL-E1 IMPROVED ACCURACY TRAINING           ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers"
echo "Skip connections: $ENABLE_SKIP"
echo "Strong anti-repetition: $STRONG_ANTI_REP"
echo "Training for $EPOCHS epochs with conservative parameters"
echo "Data format: $(if [ "$JSON_FORMAT" = true ]; then echo "JSON (max $MAX_STORIES stories)"; else echo "Plain text"; fi)"
echo "╔═════════════════════════════════════════════════════╗"
echo "║    Starting training with optimized parameters      ║"
echo "╚═════════════════════════════════════════════════════╝"

# Run the command
echo "$CMD"
$CMD

# Check result
if [ $? -eq 0 ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║       IMPROVED ACCURACY TRAINING COMPLETE           ║"
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