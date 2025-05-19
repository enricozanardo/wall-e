#!/bin/bash

# Script to run a profiled training session and format the output

# Set up output directory
PROFILE_DIR="./profiling_results"
mkdir -p "$PROFILE_DIR"

# Timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$PROFILE_DIR/profile_log_$TIMESTAMP.txt"
METRICS_FILE="$PROFILE_DIR/metrics_$TIMESTAMP.txt"

# Default values
STORIES=200
MODEL_DIM=128
FF_DIM=512
LAYERS=2
EPOCHS=1
CPUS=""
MEMORY_OPT="--use-memory-opt"
CHECKPOINT_STRATEGY=""
THREAD_OPT=""
BATCH_SIZE=""
CURRICULUM_EXAMPLES=""
DATA_FILE=""
# New parameter defaults
WATCHDOG_TIMEOUT=""
DATA_THREADS=""
TARGET_ID_MAX=""
AUTO_RESIZE_VOCAB=""
VOCAB_SIZE=""
MIN_FREQ=""
ENABLE_SKIP=""
STRONG_ANTI_REP=""
JSON_FORMAT="--json-format" # Default to JSON format since most training data is JSON

# Process arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --stories)
      STORIES="$2"
      shift 2
      ;;
    --model-dim)
      MODEL_DIM="$2"
      shift 2
      ;;
    --ff-dim)
      FF_DIM="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --cpus)
      CPUS="--cpus $2"
      shift 2
      ;;
    --use-memory-opt)
      MEMORY_OPT="--use-memory-opt"
      shift 1
      ;;
    --memory-opt)
      MEMORY_OPT="--use-memory-opt"
      shift 1
      ;;
    --checkpoint-strategy)
      CHECKPOINT_STRATEGY="--checkpoint-strategy $2"
      shift 2
      ;;
    --thread-opt)
      THREAD_OPT="--thread-opt $2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="--batch-size $2"
      shift 2
      ;;
    --curriculum-examples)
      CURRICULUM_EXAMPLES="--curriculum-examples $2"
      shift 2
      ;;
    # New parameter handling
    --watchdog-timeout)
      WATCHDOG_TIMEOUT="--watchdog-timeout $2"
      shift 2
      ;;
    --data-threads)
      DATA_THREADS="--data-threads $2"
      shift 2
      ;;
    --target-id-max)
      TARGET_ID_MAX="--target-id-max $2"
      shift 2
      ;;
    --auto-resize-vocab)
      AUTO_RESIZE_VOCAB="--auto-resize-vocab"
      shift 1
      ;;
    --vocab-size)
      VOCAB_SIZE="--vocab-size $2"
      shift 2
      ;;
    --min-freq)
      MIN_FREQ="--min-freq $2"
      shift 2
      ;;
    --enable-skip)
      ENABLE_SKIP="--enable-skip"
      shift 1
      ;;
    --strong-anti-rep)
      STRONG_ANTI_REP="--strong-anti-rep"
      shift 1
      ;;
    --json-format)
      JSON_FORMAT="--json-format"
      shift 1
      ;;
    *)
      # Assume last argument is the data file
      if [[ $# -eq 1 ]]; then
        DATA_FILE="$1"
      else
        echo "Unknown option: $1"
        exit 1
      fi
      shift 1
      ;;
  esac
done

# Ensure DATA_FILE is set
if [[ -z "$DATA_FILE" ]]; then
  echo "Error: No data file specified"
  exit 1
fi

# Ensure the data file exists
if [[ ! -f "$DATA_FILE" ]]; then
  echo "Error: Data file not found: $DATA_FILE"
  exit 1
fi

# Check if the file is empty or not valid JSON
if [[ ! -s "$DATA_FILE" ]] || ! grep -q '{' "$DATA_FILE"; then
  echo "Warning: Data file is empty or not valid JSON. Creating a minimal sample for testing."
  SAMPLE_DATA='{
    "stories": [
      {
        "text": "Once upon a time there was a little girl. She loved to play in the garden. One day she found a small box. Inside was a shiny key. She wondered what it could open. She tried it on many doors. Finally, it opened a small door to a magical world. The end."
      },
      {
        "text": "Tom had a red ball. He liked to throw it very high. One day the ball got stuck in a tree. Tom was sad. His dad helped him get it down. Tom was happy again and promised to be more careful next time."
      },
      {
        "text": "Max the dog loved to run. He ran in the park every day. One morning he saw a cat. He chased the cat up a tree. The cat was scared. Max barked and wagged his tail. Finally, the cat came down and they became friends."
      }
    ]
  }'
  echo "$SAMPLE_DATA" > "$DATA_FILE"
  echo "Created sample data with 3 short stories in $DATA_FILE"
fi

# Run the model training with profiling enabled
# Use a small dataset and few epochs to make it quick but still representative
echo "Starting profiled training run at $(date)"
echo "Results will be saved to $LOGFILE"
echo "Using data file: $DATA_FILE"

# Enhanced profiling settings
export RUST_BACKTRACE=1

# Build command arguments
CMD_ARGS=(
  "--stories" "$STORIES"
  "--model-dim" "$MODEL_DIM"
  "--ff-dim" "$FF_DIM"
  "--layers" "$LAYERS"
  "--epochs" "$EPOCHS"
  "--perf-log" "true"
  "--save-path" "$PROFILE_DIR/model_$TIMESTAMP.json"
)

# Add JSON format parameter (if set)
if [[ -n "$JSON_FORMAT" ]]; then
  CMD_ARGS+=($JSON_FORMAT)
fi

# Add optional arguments
if [[ -n "$CPUS" ]]; then
  CMD_ARGS+=($CPUS)
fi

if [[ -n "$MEMORY_OPT" ]]; then
  CMD_ARGS+=($MEMORY_OPT)
fi

if [[ -n "$BATCH_SIZE" ]]; then
  CMD_ARGS+=($BATCH_SIZE)
fi

if [[ -n "$CURRICULUM_EXAMPLES" ]]; then
  CMD_ARGS+=($CURRICULUM_EXAMPLES)
fi

if [[ -n "$CHECKPOINT_STRATEGY" ]]; then
  CMD_ARGS+=($CHECKPOINT_STRATEGY)
fi

if [[ -n "$THREAD_OPT" ]]; then
  CMD_ARGS+=($THREAD_OPT)
fi

# Add new parameters
if [[ -n "$WATCHDOG_TIMEOUT" ]]; then
  CMD_ARGS+=($WATCHDOG_TIMEOUT)
fi

if [[ -n "$DATA_THREADS" ]]; then
  CMD_ARGS+=($DATA_THREADS)
fi

if [[ -n "$TARGET_ID_MAX" ]]; then
  CMD_ARGS+=($TARGET_ID_MAX)
fi

if [[ -n "$AUTO_RESIZE_VOCAB" ]]; then
  CMD_ARGS+=($AUTO_RESIZE_VOCAB)
fi

if [[ -n "$VOCAB_SIZE" ]]; then
  CMD_ARGS+=($VOCAB_SIZE)
fi

if [[ -n "$MIN_FREQ" ]]; then
  CMD_ARGS+=($MIN_FREQ)
fi

if [[ -n "$ENABLE_SKIP" ]]; then
  CMD_ARGS+=($ENABLE_SKIP)
fi

if [[ -n "$STRONG_ANTI_REP" ]]; then
  CMD_ARGS+=($STRONG_ANTI_REP)
fi

# Add data file as the last argument
CMD_ARGS+=("$DATA_FILE")

echo "Running with parameters: ${CMD_ARGS[@]}"

# Run the training with our enhanced profiling
cargo run --release --bin Wall-E -- "${CMD_ARGS[@]}" 2>&1 | tee "$LOGFILE"

# Check if training was successful
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "Training failed! Check $LOGFILE for details."
  exit 1
fi

# Extract the performance metrics section and format it nicely
echo "Extracting performance metrics..."
sed -n '/PERFORMANCE METRICS/,/═══════════════════/p' "$LOGFILE" > "$METRICS_FILE"

echo "------------------------------------"
echo "TOP TIME-CONSUMING OPERATIONS:"
grep -A 20 "TOP TIME-CONSUMING OPERATIONS" "$METRICS_FILE" | grep "^#" 

echo "------------------------------------"
echo "PARALLELISM METRICS:"
grep -A 20 "PARALLELISM METRICS" "$METRICS_FILE" | grep -v "PARALLELISM METRICS" | grep -v "^$" | grep -v "DETAILED"

echo "------------------------------------"
echo "Performance profile completed. Results saved to:"
echo "  - Raw log: $LOGFILE"
echo "  - Metrics: $METRICS_FILE"

# Create a summary file
SUMMARY_FILE="$PROFILE_DIR/summary_$TIMESTAMP.txt"
{
  echo "===== WALL-E1 TRAINING PROFILE SUMMARY ====="
  echo "Date: $(date)"
  echo ""
  echo "CONFIGURATION:"
  echo "  Model size: ${MODEL_DIM}d (${LAYERS} layers)"
  echo "  Stories: $STORIES"
  echo "  Epochs: $EPOCHS"
  echo "  Memory optimization: $(if [[ -n "$MEMORY_OPT" ]]; then echo "enabled"; else echo "disabled"; fi)"
  echo ""
  echo "TIMING SUMMARY:"
  grep "Total batch preparation time" "$LOGFILE" || echo "  Batch preparation: Data not available"
  grep "Total training step time" "$LOGFILE" || echo "  Training step time: Data not available"
  grep "Training completed in" "$LOGFILE" || echo "  Total training time: Data not available"
  echo ""
  echo "TOP 5 TIME-CONSUMING OPERATIONS:"
  grep -A 5 "TOP TIME-CONSUMING OPERATIONS" "$METRICS_FILE" | grep "^#" || echo "  No data available"
} > "$SUMMARY_FILE"

echo "------------------------------------"
echo "Training summary saved to: $SUMMARY_FILE"
echo "To view detailed metrics, check: $METRICS_FILE" 