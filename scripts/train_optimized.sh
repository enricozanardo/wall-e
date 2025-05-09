#!/bin/bash

# Train enhanced model with optimized parameters
# This script runs the enhanced training with the best parameters for wall-e1

# Display help if requested
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "Usage: ./train_optimized.sh [training_data_path] [options]"
  echo "Default training data: data/corpus.txt"
  echo ""
  echo "Options:"
  echo "  --quick       Use smaller model for quicker training"
  echo "  --full        Use full-sized model for best results (slower)"
  echo "  --no-skip     Disable skip connections"
  echo "  --no-anti-rep Disable strong anti-repetition mechanisms"
  echo "  --save PATH   Specify custom save path"
  exit 0
fi

# Default settings
TRAINING_DATA="${1:-data/corpus.txt}"
QUICK_MODE=false
ENABLE_SKIP=true
STRONG_ANTI_REP=true
SAVE_PATH="model_enhanced.json"

# Parse command line arguments
shift 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --quick)
      QUICK_MODE=true
      shift
      ;;
    --full)
      QUICK_MODE=false
      shift
      ;;
    --no-skip)
      ENABLE_SKIP=false
      shift
      ;;
    --no-anti-rep)
      STRONG_ANTI_REP=false
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

# Check if training data exists
if [ ! -f "$TRAINING_DATA" ]; then
  echo "Error: Training data file '$TRAINING_DATA' not found!"
  echo "Please provide a valid path to training data."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p $(dirname "$SAVE_PATH")

# Set model parameters based on mode
if [ "$QUICK_MODE" = true ]; then
  echo "Using quick training mode with smaller model"
  MODEL_DIM=128
  FF_DIM=512
  HEADS=4
  LAYERS=3
  EPOCHS=5
  BATCH_SIZE=16
else
  echo "Using full training mode with larger model"
  MODEL_DIM=192
  FF_DIM=768
  HEADS=6
  LAYERS=4
  EPOCHS=10
  BATCH_SIZE=32
fi

# Build command
CMD="cargo run --release --bin train_enhanced_model -- $TRAINING_DATA"
CMD="$CMD --model-dim $MODEL_DIM"
CMD="$CMD --ff-dim $FF_DIM"
CMD="$CMD --heads $HEADS"
CMD="$CMD --layers $LAYERS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --save-path $SAVE_PATH"

# Add skip connections if enabled
if [ "$ENABLE_SKIP" = true ]; then
  CMD="$CMD --enable-skip"
fi

# Add strong anti-repetition if enabled
if [ "$STRONG_ANTI_REP" = true ]; then
  CMD="$CMD --strong-anti-rep"
fi

# Print configuration
echo "╔════════════════════════════════════════╗"
echo "║        Wall-E1 Enhanced Training       ║"
echo "╚════════════════════════════════════════╝"
echo "Training data:      $TRAINING_DATA"
echo "Model dimensions:   $MODEL_DIM"
echo "FF dimensions:      $FF_DIM"
echo "Attention heads:    $HEADS"
echo "Layers:             $LAYERS"
echo "Skip connections:   $ENABLE_SKIP"
echo "Anti-repetition:    $STRONG_ANTI_REP"
echo "Save path:          $SAVE_PATH"
echo "╔════════════════════════════════════════╗"
echo "║           Starting training...         ║"
echo "╚════════════════════════════════════════╝"

# Run the command
echo "$CMD"
eval "$CMD"

# Final message
if [ $? -eq 0 ]; then
  echo "╔════════════════════════════════════════╗"
  echo "║       Training completed successfully  ║"
  echo "║                                        ║"
  echo "║ Model saved to: $SAVE_PATH"
  echo "╚════════════════════════════════════════╝"
else
  echo "╔════════════════════════════════════════╗"
  echo "║            Training failed!            ║"
  echo "╚════════════════════════════════════════╝"
fi 