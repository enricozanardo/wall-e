#!/bin/bash
# Comprehensive training script for Wall-E1 language model
# This script implements all improvements from PDCA analysis with data preprocessing

echo "===== WALL-E1 COMPREHENSIVE TRAINING SCRIPT ====="
echo "This script combines all training improvements with data preprocessing"

# Create models directory if it doesn't exist
mkdir -p models
mkdir -p data/processed

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Default parameters - Balanced model architecture based on PDCA analysis
MODEL_DIM=128
FF_DIM=512
HEADS=4
LAYERS=3
VOCAB_SIZE=4000
MIN_FREQ=3
SAVE_PATH="models/comprehensive_model.json"
ENABLE_SKIP=true
STRONG_ANTI_REP=true
ENABLE_PREPROCESSING=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --model-dim)
      MODEL_DIM="$2"
      shift 2
      ;;
    --model-size)
      case "$2" in
        "small")
          MODEL_DIM=96
          FF_DIM=384
          HEADS=3
          LAYERS=3
          ;;
        "medium")
          MODEL_DIM=128
          FF_DIM=512
          HEADS=4
          LAYERS=3
          ;;
        "large")
          MODEL_DIM=192
          FF_DIM=768
          HEADS=6
          LAYERS=4
          ;;
        *)
          echo "Unknown model size: $2. Using medium."
          ;;
      esac
      shift 2
      ;;
    --no-skip)
      ENABLE_SKIP=false
      shift
      ;;
    --no-anti-rep)
      STRONG_ANTI_REP=false
      shift
      ;;
    --no-preprocess)
      ENABLE_PREPROCESSING=false
      shift
      ;;
    --save)
      SAVE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      shift
      ;;
  esac
done

# Default to TinyStories data if not specified
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

# STEP 1: DATA PREPROCESSING
echo "╔═════════════════════════════════════════════════════╗"
echo "║         STEP 1: DATA PREPROCESSING                  ║"
echo "╚═════════════════════════════════════════════════════╝"

PROCESSED_DATA="data/processed/cleaned_$(basename $DATA_PATH)"

if [ "$ENABLE_PREPROCESSING" = true ]; then
  echo "Preprocessing data for better tokenization and training..."

  # Create Python preprocessing script
  cat > data/preprocess.py << 'EOL'
import json
import re
import sys
from pathlib import Path

def clean_text(text):
    # Normalize spacing around punctuation
    text = re.sub(r'([.,!?:;])', r' \1 ', text)
    # Fix contractions
    text = re.sub(r"(\w)'(\w)", r"\1' \2", text)  
    # Normalize quotes - use simpler approach
    text = re.sub(r'"', ' " ', text)
    text = re.sub(r"'", " ' ", text)
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stories = data.get('stories', [])
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
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if input_path.endswith('.json'):
        process_json_file(input_path, output_path)
    else:
        process_text_file(input_path, output_path)
EOL

  # Run the preprocessing - use python3 explicitly
  python3 data/preprocess.py "$DATA_PATH" "$PROCESSED_DATA" || {
    echo "Error: Data preprocessing failed! Using original data."
    PROCESSED_DATA="$DATA_PATH"
  }
else
  echo "Preprocessing disabled. Using original data."
  PROCESSED_DATA="$DATA_PATH"
fi

if [ ! -f "$PROCESSED_DATA" ]; then
  echo "Error: Processed data file not found! Using original data."
  PROCESSED_DATA="$DATA_PATH"
fi

# STEP 2: CURRICULUM LEARNING IMPROVEMENT
echo "╔═════════════════════════════════════════════════════╗"
echo "║     STEP 2: CURRICULUM LEARNING IMPROVEMENT         ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Applying improvements to curriculum learning..."

# Run the enhanced training with our improved curriculum learning
echo "Building command with optimized parameters..."
CMD="cargo run --release --bin train_enhanced_model -- $PROCESSED_DATA"
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs 15" # Increased epochs
CMD="$CMD --vocab-size $VOCAB_SIZE"
CMD="$CMD --min-freq $MIN_FREQ"
CMD="$CMD --save-path $SAVE_PATH"
CMD="$CMD --learning-rate 0.0002" # Better learning rate for stable gradient flow

# Add JSON format parameters if needed
if [ "$JSON_FORMAT" = true ]; then
  CMD="$CMD --json-format --stories 8000"
fi

# Add skip connections if enabled
if [ "$ENABLE_SKIP" = true ]; then
  CMD="$CMD --enable-skip"
fi

# Add anti-repetition if enabled
if [ "$STRONG_ANTI_REP" = true ]; then
  CMD="$CMD --strong-anti-rep"
fi

# Print configuration information
echo "╔═════════════════════════════════════════════════════╗"
echo "║        COMPREHENSIVE TRAINING CONFIGURATION         ║"
echo "╚═════════════════════════════════════════════════════╝"
echo "Model architecture: $MODEL_DIM dim, $HEADS heads, $LAYERS layers"
echo "Feed-forward dimension: $FF_DIM"
echo "Skip connections: $ENABLE_SKIP"
echo "Strong anti-repetition: $STRONG_ANTI_REP"
echo "Data preprocessing: $ENABLE_PREPROCESSING"
echo "Training data: $PROCESSED_DATA"
echo "Save path: $SAVE_PATH"
echo "╔═════════════════════════════════════════════════════╗"
echo "║           STARTING ENHANCED TRAINING                ║"
echo "╚═════════════════════════════════════════════════════╝"

# Run the training command
echo "Executing: $CMD"
$CMD

# Check result
if [ $? -eq 0 ]; then
  echo "╔═════════════════════════════════════════════════════╗"
  echo "║     COMPREHENSIVE TRAINING COMPLETED SUCCESSFULLY   ║"
  echo "║                                                     ║"
  echo "║  Model saved to: $SAVE_PATH"
  echo "╚═════════════════════════════════════════════════════╝"
  
  # Generate sample text to demonstrate model improvement
  echo "Generating sample text to demonstrate model quality..."
  PROMPTS=("Once upon a time" "The quick brown fox" "In a world where" "Today I will")
  
  for PROMPT in "${PROMPTS[@]}"; do
    echo "Prompt: \"$PROMPT\""
    cargo run --release --bin train_enhanced_model -- --model $SAVE_PATH --prompt "$PROMPT" --max-tokens 50 --generate-only
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