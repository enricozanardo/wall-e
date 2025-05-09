#!/bin/bash
# Advanced training script based on PDCA analysis results
# Implements a two-phase approach with pretraining for better word boundary learning

echo "===== WALL-E1 ADVANCED TRAINING SCRIPT ====="
echo "This script implements a two-phase training approach with better data handling"

# Create models directory if it doesn't exist
mkdir -p models

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Default parameters - Refined model architecture based on analysis
MODEL_DIM=128      # Increased from 96 based on analysis
FF_DIM=512         # Increased from 384 for better capacity
HEADS=4            # Increased from 3 for better attention
LAYERS=3           # Keeping 3 layers for efficiency
EPOCHS_PHASE1=5    # Phase 1: Word boundary learning
EPOCHS_PHASE2=10   # Phase 2: Full text generation
VOCAB_SIZE=4000    # Keeping smaller vocab size that worked well
SAVE_PATH="models/advanced_model.json"
ENABLE_SKIP=true   # Enable skip connections for better gradient flow
STRONG_ANTI_REP=true # Enable improved anti-repetition for phase 2
MIN_FREQ=3         # Keep minimum token frequency at 3
JSON_FORMAT=true
MAX_STORIES=8000   # Process up to this many stories from JSON dataset
PRETRAINING=true   # Enable pretraining phase focused on word boundaries
CLEAN_DATA=false   # Whether to apply preprocessing for cleaner data

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-pretraining)
      PRETRAINING=false
      shift
      ;;
    --clean-data)
      CLEAN_DATA=true
      shift
      ;;
    --model-dim)
      MODEL_DIM="$2"
      shift 2
      ;;
    --ff-dim)
      FF_DIM="$2"
      shift 2
      ;;
    --heads)
      HEADS="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --vocab-size)
      VOCAB_SIZE="$2"
      shift 2
      ;;
    --save-path)
      SAVE_PATH="$2"
      shift 2
      ;;
    *)
      # Default assume it's the data path
      DATA_PATH="$1"
      shift
      ;;
  esac
done

# Default to TinyStories data
DATA_PATH="${DATA_PATH:-data/tiny_stories_sample.json}"
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

# Pre-process data if requested
PROCESSED_DATA="$DATA_PATH"
if [ "$CLEAN_DATA" = true ]; then
  echo "Preprocessing data for cleaner training..."
  PROCESSED_DATA="data/processed_$(basename $DATA_PATH)"
  
  # Extract and clean text
  if [ "$JSON_FORMAT" = true ]; then
    # Process JSON data
    python -c "
import json, re, sys
with open('$DATA_PATH', 'r') as f:
    data = json.load(f)
stories = data.get('stories', [])
cleaned = []
for story in stories:
    if isinstance(story, str):
        text = story
    elif isinstance(story, dict) and 'text' in story:
        text = story['text']
    else:
        continue
    # Clean spacing around punctuation
    text = re.sub(r'([.,!?])', r' \1 ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    cleaned.append(text)
with open('$PROCESSED_DATA', 'w') as f:
    json.dump({'stories': cleaned}, f)
print(f'Processed {len(cleaned)} stories with cleaned punctuation and spacing')
"
  else
    # Process plain text data
    python -c "
import re
with open('$DATA_PATH', 'r') as f:
    text = f.read()
# Clean spacing around punctuation
text = re.sub(r'([.,!?])', r' \1 ', text)
text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
with open('$PROCESSED_DATA', 'w') as f:
    f.write(text)
print(f'Processed text with cleaned punctuation and spacing')
"
  fi
  
  echo "Preprocessing complete. Using processed data at $PROCESSED_DATA"
fi

echo "╔═════════════════════════════════════════════════════╗"
echo "║         WALL-E1 ADVANCED TRAINING SETUP             ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers"
echo "Skip connections: $ENABLE_SKIP"
echo "Two-phase training: $PRETRAINING"
echo "Clean data processing: $CLEAN_DATA"
echo "Training data: $PROCESSED_DATA"

# Phase 1: Word boundary pretraining (if enabled)
if [ "$PRETRAINING" = true ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║    PHASE 1: WORD BOUNDARY PRETRAINING               ║"
  echo "╚═════════════════════════════════════════════════════╝"
  
  # For pretraining, focus just on word boundaries without strong anti-repetition
  # We'll use a small learning rate and only a few epochs
  PRETRAINING_MODEL="models/pretraining_model.json"
  
  # Build pretraining command
  CMD="cargo run --release --bin train_enhanced_model -- $PROCESSED_DATA"
  CMD="$CMD --model-dim $MODEL_DIM"
  CMD="$CMD --ff-dim $FF_DIM"
  CMD="$CMD --heads $HEADS"
  CMD="$CMD --layers $LAYERS"
  CMD="$CMD --epochs $EPOCHS_PHASE1"
  CMD="$CMD --vocab-size $VOCAB_SIZE"
  CMD="$CMD --save-path $PRETRAINING_MODEL"
  CMD="$CMD --min-freq $MIN_FREQ"
  CMD="$CMD --learning-rate 0.0001" # Lower learning rate for pretraining
  
  # Add JSON format parameters if needed
  if [ "$JSON_FORMAT" = true ]; then
    CMD="$CMD --json-format --stories $MAX_STORIES"
  fi
  
  # Add skip connections if enabled
  if [ "$ENABLE_SKIP" = true ]; then
    CMD="$CMD --enable-skip"
  fi
  
  # Run pretraining (focus on word boundaries)
  echo "Running pretraining phase to learn word boundaries..."
  echo "$CMD"
  $CMD
  
  if [ $? -ne 0 ]; then
    echo "╔═════════════════════════════════════════════════════╗"
    echo "║        PRETRAINING PHASE FAILED                     ║"
    echo "╚═════════════════════════════════════════════════════╝"
    echo "Error in pretraining phase. See above for details."
    exit 1
  fi
  
  echo "Pretraining phase completed successfully."
fi

# Phase 2: Full training with anti-repetition
echo "╔═════════════════════════════════════════════════════╗"
echo "║    PHASE 2: FULL MODEL TRAINING                     ║"
echo "╚═════════════════════════════════════════════════════╝"

# Build main training command
CMD="cargo run --release --bin train_enhanced_model -- $PROCESSED_DATA"
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs $EPOCHS_PHASE2"
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --save-path $SAVE_PATH"
CMD="$CMD --min-freq $MIN_FREQ"
CMD="$CMD --learning-rate 0.0002" # Moderate learning rate for main training

# Add initialization from pretraining if it was used
if [ "$PRETRAINING" = true ] && [ -f "$PRETRAINING_MODEL" ]; then
  CMD="$CMD --init-from $PRETRAINING_MODEL" # Load weights from pretraining model
fi

# Add JSON format parameters if needed
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

# Run main training phase
echo "Running main training phase with advanced parameters..."
echo "$CMD"
$CMD

# Check result
if [ $? -eq 0 ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║       ADVANCED TRAINING COMPLETE                    ║"
  echo "║                                                     ║"
  echo "║  Model saved to: $SAVE_PATH"
  echo "╚═════════════════════════════════════════════════════╝"
  echo "You can test the model with:"
  echo "cargo run --release --bin generate -- --model $SAVE_PATH --prompt \"Once upon a time\""
else
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║            TRAINING FAILED                          ║"
  echo "╚═════════════════════════════════════════════════════╝"
  echo "Check the error messages above for details."
fi 