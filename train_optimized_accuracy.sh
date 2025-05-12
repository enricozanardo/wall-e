#!/bin/bash
# Optimized training script for Wall-E1 language model
# This script focuses on achieving higher accuracy with reasonable training time

# Create log directory if needed
mkdir -p logs

# Default log file
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

echo "╔═════════════════════════════════════════════════════╗"
echo "║      WALL-E1 OPTIMIZED ACCURACY TRAINING SCRIPT     ║"
echo "║                                                     ║"
echo "║      Focusing on achieving higher accuracy with     ║"
echo "║            balanced training parameters             ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Logging all output to: $LOG_FILE"

# Create models directory if it doesn't exist
mkdir -p models
mkdir -p data/processed

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Parse arguments
SAVE_PATH="models/high_accuracy_model.json"
DATA_SAMPLING="balanced"  # balanced, quality, or quantity
VOCAB_SIZE=5000          # Larger vocabulary for better token differentiation
MIN_FREQ=2               # Lower threshold to include more rare words
LEARNING_RATE=0.0001     # Lower learning rate for more precise convergence
EPOCHS=20                # More epochs for better convergence
MODEL_SIZE="medium"      # Default to medium size
ENABLE_PREPROCESSING=false  # Disable preprocessing by default

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --model-size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --save)
      SAVE_PATH="$2"
      shift 2
      ;;
    --preprocess)
      ENABLE_PREPROCESSING=true
      shift
      ;;
    --data-sampling)
      DATA_SAMPLING="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Set up logging - all output will be directed to both the terminal and the log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Set model size parameters
case "$MODEL_SIZE" in
  "small")
    MODEL_DIM=128        # Increased from default small (96)
    FF_DIM=512
    HEADS=4
    LAYERS=3
    ;;
  "medium")
    MODEL_DIM=192
    FF_DIM=768
    HEADS=6
    LAYERS=4
    ;;
  "large")
    MODEL_DIM=256
    FF_DIM=1024
    HEADS=8
    LAYERS=5
    ;;
  *)
    echo "Unknown model size: $MODEL_SIZE. Using medium."
    MODEL_DIM=192
    FF_DIM=768
    HEADS=6
    LAYERS=4
    ;;
esac

# Default to TinyStories data if not specified
DATA_PATH="${DATA_PATH:-data/tiny_stories_sample.json}"
if [ ! -f "$DATA_PATH" ]; then
  # Try alternate data files
  ALTERNATES=("data/tiny_stories.json" "data/tiny_stories_sample_updated.json" "data/story.txt" "data/qa_en_dataset.json")
  
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

# Set data sampling parameters based on strategy
STORIES_COUNT=8000  # Default
case "$DATA_SAMPLING" in
  "balanced")
    STORIES_COUNT=8000
    echo "Using balanced data sampling: 8000 stories"
    ;;
  "quality")
    STORIES_COUNT=4000
    echo "Using quality-focused sampling: 4000 high-quality stories"
    ;;
  "quantity")
    STORIES_COUNT=12000
    echo "Using quantity-focused sampling: 12000 stories"
    ;;
  *)
    echo "Unknown data sampling strategy: $DATA_SAMPLING. Using balanced (8000 stories)."
    ;;
esac

# STEP 1: DATA PREPROCESSING (if enabled)
echo "╔═════════════════════════════════════════════════════╗"
echo "║              STEP 1: DATA PREPARATION               ║"
echo "╚═════════════════════════════════════════════════════╝"

PROCESSED_DATA="$DATA_PATH"
if [ "$ENABLE_PREPROCESSING" = true ]; then
  echo "Preprocessing data for better tokenization and training..."
  PROCESSED_DATA="data/processed/cleaned_$(basename $DATA_PATH)"
  
  # Create Python preprocessing script with improved cleaning
  cat > data/preprocess.py << 'EOL'
import json
import re
import sys
import random
from pathlib import Path

def clean_text(text):
    # Normalize spacing around punctuation
    text = re.sub(r'([.,!?:;])', r' \1 ', text)
    # Fix contractions
    text = re.sub(r"(\w)'(\w)", r"\1' \2", text)  
    # Normalize quotes - use simpler approach
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r"'", " ' ", text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Ensure space after periods for better sentence segmentation
    text = re.sub(r'\.([A-Z])', r'. \1', text)
    return text.strip()

def process_json_file(input_path, output_path, max_stories=8000, quality_filter=False):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stories = data.get('stories', [])
    print(f"Found {len(stories)} stories in JSON file")
    
    if quality_filter:
        # Apply quality filtering (e.g., longer stories tend to be higher quality)
        stories = [s for s in stories if isinstance(s, dict) and 'text' in s and len(s['text']) > 500]
        # Sort by length (proxy for quality)
        stories.sort(key=lambda x: len(x['text']) if isinstance(x, dict) and 'text' in x else 0, reverse=True)
    
    # Randomly sample if we have more than max_stories
    if len(stories) > max_stories:
        stories = random.sample(stories, max_stories)
    
    cleaned_stories = []
    for story in stories:
        if isinstance(story, str):
            text = story
        elif isinstance(story, dict) and 'text' in story:
            text = story['text']
        else:
            continue
        
        cleaned_text = clean_text(text)
        cleaned_stories.append(cleaned_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'stories': cleaned_stories}, f)
    
    print(f'Processed {len(cleaned_stories)} stories')
    return len(cleaned_stories)

def process_text_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned_text = clean_text(text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f'Processed {len(cleaned_text)} characters of text')
    return len(cleaned_text)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py <input_file> <output_file> [max_stories] [quality_filter]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_stories = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    quality_filter = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False
    
    if input_path.endswith('.json'):
        process_json_file(input_path, output_path, max_stories, quality_filter)
    else:
        process_text_file(input_path, output_path)
EOL

  # Run the preprocessing - use python3 explicitly
  QUALITY_FILTER="false"
  if [ "$DATA_SAMPLING" = "quality" ]; then
    QUALITY_FILTER="true"
  fi
  
  python3 data/preprocess.py "$DATA_PATH" "$PROCESSED_DATA" "$STORIES_COUNT" "$QUALITY_FILTER" || {
    echo "Error: Data preprocessing failed! Using original data."
    PROCESSED_DATA="$DATA_PATH"
  }
else
  echo "Preprocessing disabled. Using original data."
fi

# Print configuration information
echo "╔═════════════════════════════════════════════════════╗"
echo "║          OPTIMIZED TRAINING CONFIGURATION           ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Model size: $MODEL_SIZE"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers"
echo "Feed-forward dimension: $FF_DIM"
echo "Vocabulary size: $VOCAB_SIZE"
echo "Min token frequency: $MIN_FREQ"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Data sampling: $DATA_SAMPLING ($STORIES_COUNT stories)"
echo "Data preprocessing: $ENABLE_PREPROCESSING"
echo "Training data: $PROCESSED_DATA"
echo "Save path: $SAVE_PATH"

# STEP 2: BUILD TRAINING COMMAND WITH OPTIMIZED PARAMETERS
echo "╔═════════════════════════════════════════════════════╗"
echo "║           BUILDING OPTIMIZED COMMAND                ║"
echo "╚═════════════════════════════════════════════════════╝"

# Build command with our optimized parameters
CMD="cargo run --release --bin train_enhanced_model -- $PROCESSED_DATA"
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --min-freq $MIN_FREQ"
CMD="$CMD --save-path $SAVE_PATH"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --enable-skip"      # Always enable skip connections for better gradient flow
CMD="$CMD --strong-anti-rep"  # Always use strong anti-repetition

# Add JSON format parameters if needed
if [ "$JSON_FORMAT" = true ]; then
  CMD="$CMD --json-format --stories $STORIES_COUNT"
fi

# STEP 3: RUN THE TRAINING
echo "╔═════════════════════════════════════════════════════╗"
echo "║           STARTING OPTIMIZED TRAINING               ║"
echo "╚═════════════════════════════════════════════════════╝"

echo "Executing: $CMD"
$CMD

# Check result
if [ $? -eq 0 ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║       OPTIMIZED ACCURACY TRAINING COMPLETE!         ║"
  echo "║                                                     ║"
  echo "║  Model saved to: $SAVE_PATH"
  echo "╚═════════════════════════════════════════════════════╝"
  
  # Generate sample text
  echo "Generating sample text to demonstrate model quality..."
  PROMPTS=("Once upon a time" "The quick brown fox" "In a world where magic" "The most important thing")
  
  for PROMPT in "${PROMPTS[@]}"; do
    echo "Prompt: \"$PROMPT\""
    cargo run --release --bin train_enhanced_model -- --model $SAVE_PATH --prompt "$PROMPT" --max-tokens 75 --generate-only
    echo ""
  done
  
  echo "Training complete. You can test the model with:"
  echo "cargo run --release --bin train_enhanced_model -- --model $SAVE_PATH --prompt \"Your prompt here\" --generate-only"
else
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║           TRAINING FAILED                           ║"
  echo "╚═════════════════════════════════════════════════════╝"
  echo "Check the error messages above for details."
fi 