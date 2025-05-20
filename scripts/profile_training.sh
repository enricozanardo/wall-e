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
PARALLEL_DATA_PREP="" # New parameter for parallel data preparation
MT_TRAINING=""

# Help message function
show_usage() {
    echo "Usage: $0 [options] <data_file>"
    echo "Runs training with performance profiling enabled"
    echo ""
    echo "Options:"
    echo "  --stories <number>           Number of stories to process (default: $STORIES)"
    echo "  --model-dim <number>         Model dimension (default: $MODEL_DIM)"
    echo "  --ff-dim <number>            Feed-forward dimension (default: $FF_DIM)"
    echo "  --layers <number>            Number of layers (default: $LAYERS)"
    echo "  --epochs <number>            Number of epochs (default: $EPOCHS)"
    echo "  --cpus <number>              Number of CPUs to use"
    echo "  --memory-opt                 Enable memory optimization (default: enabled)"
    echo "  --no-memory-opt              Disable memory optimization"
    echo "  --checkpoint <strategy>      Checkpoint strategy (uniform, layerwise, adaptive)"
    echo "  --thread-opt <strategy>      Thread optimization strategy (default, aggressive, conservative)"
    echo "  --batch-size <size>          Batch size"
    echo "  --curriculum-examples <num>  Number of curriculum examples"
    echo "  --auto-resize-vocab          Automatically resize vocabulary"
    echo "  --vocab-size <number>        Vocabulary size"
    echo "  --min-freq <number>          Minimum token frequency"
    echo "  --watchdog-timeout <seconds> Watchdog timeout in seconds"
    echo "  --data-threads <number>      Number of data loading threads"
    echo "  --target-id-max <number>     Maximum target ID for tokens"
    echo "  --enable-skip                Enable skip connections"
    echo "  --strong-anti-rep            Use stronger anti-repetition"
    echo "  --json-format                Process input as JSON format"
    echo "  --parallel                  Enable parallel data preparation (for faster training)"
    echo "  --mt-training               Enable multi-threaded model training (experimental)"
    echo "  --help                       Show this help message"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        --memory-opt|--use-memory-opt)
            MEMORY_OPT="--use-memory-opt"
            shift
            ;;
        --no-memory-opt)
            MEMORY_OPT=""
            shift
            ;;
        --checkpoint)
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
        --auto-resize-vocab)
            AUTO_RESIZE_VOCAB="--auto-resize-vocab"
            shift
            ;;
        --vocab-size)
            VOCAB_SIZE="--vocab-size $2"
            shift 2
            ;;
        --min-freq)
            MIN_FREQ="--min-freq $2"
            shift 2
            ;;
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
        --enable-skip)
            ENABLE_SKIP="--enable-skip"
            shift
            ;;
        --strong-anti-rep)
            STRONG_ANTI_REP="--strong-anti-rep"
            shift
            ;;
        --json-format)
            JSON_FORMAT="--json-format"
            shift
            ;;
        --parallel-data-prep|--parallel)
            PARALLEL_DATA_PREP="--parallel"
            shift
            ;;
        --mt-training|--mt)
            MT_TRAINING="--mt-training"
            PARALLEL_DATA_PREP="--parallel"  # MT training implies parallel data prep
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            # If it's the last argument and not a flag, treat it as the data file
            if [[ $# -eq 1 && ! $1 == --* ]]; then
                DATA_FILE="$1"
                shift
            else
                echo "Unknown option: $1"
                show_usage
                exit 1
            fi
            ;;
    esac
done

# Check if data file is provided
if [[ -z "$DATA_FILE" ]]; then
    echo "Error: No data file specified"
    show_usage
    exit 1
fi

# Ensure the data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: Data file not found at: $DATA_FILE"
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

# Model save path
SAVE_PATH="models/profiled_model_${TIMESTAMP}.bin"

# Start timing
START_TIME=$(date +%s)

# Run the training command with profiling enabled
echo "Starting profiled training with log output to: $LOGFILE"
echo "Command: cargo run --release --bin train_enhanced_model -- --dataset $DATA_FILE --model-dim $MODEL_DIM --ff-dim $FF_DIM --layers $LAYERS --epochs $EPOCHS --max-stories $STORIES --save-path $SAVE_PATH $CPUS $MEMORY_OPT $CHECKPOINT_STRATEGY $THREAD_OPT $BATCH_SIZE $CURRICULUM_EXAMPLES $AUTO_RESIZE_VOCAB $VOCAB_SIZE $MIN_FREQ $WATCHDOG_TIMEOUT $DATA_THREADS $TARGET_ID_MAX $ENABLE_SKIP $STRONG_ANTI_REP $JSON_FORMAT $PARALLEL_DATA_PREP $MT_TRAINING --perf-log" | tee -a "$LOGFILE"

# Execute with timing and logging
{
    time cargo run --release --bin train_enhanced_model -- \
        --dataset "$DATA_FILE" \
        --model-dim "$MODEL_DIM" \
        --ff-dim "$FF_DIM" \
        --layers "$LAYERS" \
        --epochs "$EPOCHS" \
        --max-stories "$STORIES" \
        --save-path "$SAVE_PATH" \
        $CPUS \
        $MEMORY_OPT \
        $CHECKPOINT_STRATEGY \
        $THREAD_OPT \
        $BATCH_SIZE \
        $CURRICULUM_EXAMPLES \
        $AUTO_RESIZE_VOCAB \
        $VOCAB_SIZE \
        $MIN_FREQ \
        $WATCHDOG_TIMEOUT \
        $DATA_THREADS \
        $TARGET_ID_MAX \
        $ENABLE_SKIP \
        $STRONG_ANTI_REP \
        $JSON_FORMAT \
        $PARALLEL_DATA_PREP \
        $MT_TRAINING \
        --perf-log
} 2>&1 | tee -a "$LOGFILE"

# End timing
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Extract metrics and save to separate file
{
    echo "PROFILING SUMMARY"
    echo "================="
    echo "Date: $(date)"
    echo "Duration: $TOTAL_TIME seconds"
    echo ""
    echo "CONFIGURATION"
    echo "Model dimension: $MODEL_DIM"
    echo "Feed-forward dimension: $FF_DIM"
    echo "Layers: $LAYERS"
    echo "Epochs: $EPOCHS"
    echo "Stories: $STORIES"
    echo "Memory optimization: $(if [[ -n "$MEMORY_OPT" ]]; then echo "enabled"; else echo "disabled"; fi)"
    echo "Parallel data preparation: $(if [[ -n "$PARALLEL_DATA_PREP" ]]; then echo "enabled"; else echo "disabled"; fi)"
    echo "Multi-threaded training: $(if [[ -n "$MT_TRAINING" ]]; then echo "enabled"; else echo "disabled"; fi)"
    echo ""
    echo "EXTRACTED METRICS"
    
    # Extract training time
    TRAIN_TIME=$(grep -Eo "Completed .+ training in [0-9.]+s" "$LOGFILE" | grep -Eo "[0-9.]+s")
    echo "Training time: $TRAIN_TIME"
    
    # Extract memory usage
    MEM_USAGE=$(grep -Eo "Memory usage: [0-9.]+ MB" "$LOGFILE" | sort -nr | head -1 | grep -Eo "[0-9.]+ MB")
    echo "Peak memory usage: $MEM_USAGE"
    
    # Extract CPU utilization
    CPU_USAGE=$(grep -Eo "CPU utilization: [0-9.]+%" "$LOGFILE" | sort -nr | head -1 | grep -Eo "[0-9.]+%")
    echo "CPU utilization: $CPU_USAGE"
    
    # Extract average loss
    AVG_LOSS=$(grep -Eo "average loss: [0-9.]+" "$LOGFILE" | tail -1 | grep -Eo "[0-9.]+")
    echo "Final average loss: $AVG_LOSS"
    
    # Extract profiling data
    echo ""
    echo "OPERATION TIMING"
    grep -E "Operation .+ took [0-9.]+ seconds" "$LOGFILE" | sort -k5 -nr
} > "$METRICS_FILE"

echo ""
echo "Profiling complete!"
echo "Full log: $LOGFILE"
echo "Metrics summary: $METRICS_FILE"

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