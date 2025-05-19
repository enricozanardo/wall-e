#!/bin/bash

# Script for optimized training with accuracy focus

# Default model size
MODEL_SIZE="small"
# Default number of stories
NUM_STORIES=4000
# Default CPU cores (0 means use all available)
NUM_CPUS=0
# Memory optimization flag
MEMORY_OPT=""
# Checkpoint strategy
CHECKPOINT_STRATEGY="adaptive"
# Thread optimization
THREAD_OPT=""
# Batch size (0 means auto-calculate)
BATCH_SIZE=0
# Default number of epochs
NUM_EPOCHS=10
# Default curriculum examples
CURRICULUM_EXAMPLES=500
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
    --stories)
      NUM_STORIES="$2"
      shift 2
      ;;
    --cpus)
      NUM_CPUS="$2"
      shift 2
      ;;
    --memory-opt)
      MEMORY_OPT="true"
      shift 1
      ;;
    --checkpoint-strategy)
      CHECKPOINT_STRATEGY="$2"
      shift 2
      ;;
    --thread-opt)
      THREAD_OPT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      NUM_EPOCHS="$2"
      shift 2
      ;;
    --curriculum-examples)
      CURRICULUM_EXAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--size small|medium|large] [--stories <number>] [--cpus <number>] [--memory-opt] [--checkpoint-strategy <strategy>] [--thread-opt <operation>] [--batch-size <number>] [--epochs <number>] [--curriculum-examples <number>]"
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
echo "Using quality-focused sampling: $NUM_STORIES high-quality stories" | tee -a "$LOG_FILE"

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
echo "Epochs: $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "Data sampling: quality ($NUM_STORIES stories)" | tee -a "$LOG_FILE"
echo "Curriculum examples: $CURRICULUM_EXAMPLES" | tee -a "$LOG_FILE"
echo "Data preprocessing: false" | tee -a "$LOG_FILE"
echo "Training data: data/tiny_stories_sample.json" | tee -a "$LOG_FILE"
echo "Save path: models/high_accuracy_model.walle" | tee -a "$LOG_FILE"

# Memory optimization settings
if [ -n "$MEMORY_OPT" ]; then
  echo "Memory optimization: enabled" | tee -a "$LOG_FILE"
  
  # Show checkpoint strategy if memory optimization is enabled
  echo "Checkpoint strategy: $CHECKPOINT_STRATEGY" | tee -a "$LOG_FILE"
else
  echo "Memory optimization: disabled" | tee -a "$LOG_FILE"
fi

# Thread optimization settings
if [ -n "$THREAD_OPT" ]; then
  echo "Thread optimization: $THREAD_OPT" | tee -a "$LOG_FILE"
else
  echo "Thread optimization: auto" | tee -a "$LOG_FILE"
fi

# Batch size settings
if [ "$BATCH_SIZE" -gt 0 ]; then
  echo "Batch size: $BATCH_SIZE (manual)" | tee -a "$LOG_FILE"
else
  echo "Batch size: auto-calculated" | tee -a "$LOG_FILE"
fi

# Set CPU Cores information
if [ "$NUM_CPUS" -gt 0 ]; then
  echo "CPU Cores: $NUM_CPUS" | tee -a "$LOG_FILE"
  export RAYON_NUM_THREADS=$NUM_CPUS
else
  # Get available CPU cores
  AVAILABLE_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  echo "CPU Cores: $AVAILABLE_CPUS (all available)" | tee -a "$LOG_FILE"
  export RAYON_NUM_THREADS=$AVAILABLE_CPUS
fi

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║           BUILDING OPTIMIZED COMMAND                ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

echo "╔═════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║           STARTING OPTIMIZED TRAINING               ║" | tee -a "$LOG_FILE"
echo "╚═════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"

# Build basic training command
TRAINING_CMD="cargo run --release --bin train_enhanced_model -- data/tiny_stories_sample.json --model-dim $MODEL_DIM --ff-dim $FF_DIM --heads $HEADS --layers $LAYERS --epochs $NUM_EPOCHS --vocab-size 5000 --min-freq 2 --save-path models/high_accuracy_model.walle --learning-rate 0.0001 --enable-skip --strong-anti-rep --json-format --stories $NUM_STORIES --perf-log true --curriculum-examples $CURRICULUM_EXAMPLES"

# Add memory optimization flags if enabled
if [ -n "$MEMORY_OPT" ]; then
  TRAINING_CMD="$TRAINING_CMD --use-memory-opt"
  
  # Add checkpoint strategy if memory optimization is enabled
  if [ -n "$CHECKPOINT_STRATEGY" ]; then
    TRAINING_CMD="$TRAINING_CMD --checkpoint-strategy $CHECKPOINT_STRATEGY"
  fi
fi

# Add thread optimization if specified
if [ -n "$THREAD_OPT" ]; then
  TRAINING_CMD="$TRAINING_CMD --thread-opt $THREAD_OPT"
fi

# Add manual batch size if specified
if [ "$BATCH_SIZE" -gt 0 ]; then
  TRAINING_CMD="$TRAINING_CMD --batch-size $BATCH_SIZE"
fi

# Execute the training command
echo "Executing: $TRAINING_CMD" | tee -a "$LOG_FILE"
echo "Using RAYON_NUM_THREADS=$RAYON_NUM_THREADS" | tee -a "$LOG_FILE"
eval $TRAINING_CMD | tee -a "$LOG_FILE"

echo "Training completed. Results saved to $LOG_FILE"
echo "Model saved to models/high_accuracy_model.walle"

# Function to generate text with the trained model
generate_text() {
  local prompt="$1"
  echo "Prompt: \"$prompt\"" | tee -a "$LOG_FILE"
  
  # Note the ordering of arguments - make sure --model comes before the path
  # and generate-only is a flag without a value
  GENERATE_CMD="cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.walle --prompt \"$prompt\" --max-tokens 75 --perf-log true"
  eval $GENERATE_CMD | tee -a "$LOG_FILE"
}

echo "Generating sample text to demonstrate model quality..." | tee -a "$LOG_FILE"
generate_text "Once upon a time"
generate_text "The quick brown fox"
generate_text "In a world where magic"
generate_text "The most important thing"

echo "Training complete. You can test the model with:" | tee -a "$LOG_FILE"
echo "cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.walle --prompt \"Your prompt here\" --max-tokens 100" | tee -a "$LOG_FILE" 