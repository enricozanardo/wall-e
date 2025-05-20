#!/bin/bash

# Wall-E1 optimized training script for high accuracy

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --size small|medium|large    Model size (default: small)"
    echo "  --stories <number>           Number of stories to use (default: 4000)"
    echo "  --cpus <number>              Number of CPU cores to use"
    echo "  --memory-opt                 Enable memory optimization"
    echo "  --checkpoint-strategy <str>  Gradient checkpointing strategy: boundary, uniform, adaptive"
    echo "  --thread-opt <operation>     Thread optimization for: matrix_multiply, attention, gradient_update, data_loading"
    echo "  --batch-size <number>        Manual batch size"
    echo "  --epochs <number>            Number of training epochs (default: 10)"
    echo "  --curriculum-examples <num>  Number of curriculum examples (default: 500)"
    echo "  --auto-resize-vocab          Enable automatic vocabulary resizing"
    echo "  --watchdog-timeout <secs>    Timeout for watchdog thread detection (default: 60)"
    echo "  --batch-timeout <secs>       Timeout for batch processing in multi-threaded mode (default: 60)"
    echo "  --data-threads <number>      Number of threads for data loading"
    echo "  --target-id-max <number>     Maximum target ID value (default: auto-detected)"
    echo "  --parallel                   Enable parallel data preparation (for faster training)"
    echo "  --mt-training                Enable multi-threaded model training (experimental)"
    echo "  --help                       Display this help message"
}

# Default values
SIZE="small"
STORIES=4000
CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
MEMORY_OPT=""
CHECKPOINT_STRATEGY=""
THREAD_OPT=""
BATCH_SIZE=""
EPOCHS=10
CURRICULUM_EXAMPLES=500
AUTO_RESIZE_VOCAB=""
WATCHDOG_TIMEOUT=""
DATA_THREADS=""
TARGET_ID_MAX=""
PARALLEL_DATA_PREP=""
MT_TRAINING=""
BATCH_TIMEOUT=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --stories)
            STORIES="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --memory-opt)
            MEMORY_OPT="--memory-opt"
            shift
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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --curriculum-examples)
            CURRICULUM_EXAMPLES="$2"
            shift 2
            ;;
        --auto-resize-vocab)
            AUTO_RESIZE_VOCAB="--auto-resize-vocab"
            shift
            ;;
        --watchdog-timeout)
            WATCHDOG_TIMEOUT="--watchdog-timeout $2"
            shift 2
            ;;
        --batch-timeout)
            BATCH_TIMEOUT="--batch-timeout $2"
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
        --reliable-training)
            echo "Note: Reliable training is now the default mode, --reliable-training option is not needed"
            shift
            ;;
        --parallel-data-prep|--parallel)
            PARALLEL_DATA_PREP="--parallel"
            shift
            ;;
        --mt-training|--mt)
            MT_TRAINING="--mt-training"
            PARALLEL_DATA_PREP="--parallel"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set model dimensions based on size
case $SIZE in
    small)
        MODEL_DIM=128
        FF_DIM=512
        HEADS=4
        LAYERS=2
        ;;
    medium)
        MODEL_DIM=256
        FF_DIM=1024
        HEADS=8
        LAYERS=4
        ;;
    large)
        MODEL_DIM=512
        FF_DIM=2048
        HEADS=8
        LAYERS=8
        ;;
    *)
        echo "Invalid size: $SIZE (must be small, medium, or large)"
        exit 1
        ;;
esac

# Check for dataset
DATA_FILE="./data/tiny_stories_sample.json"

# First check for the 10k version
if [[ -f "./data/tinystories-10k.json" ]]; then
    DATA_FILE="./data/tinystories-10k.json"
elif [[ -f "./data/tinystories-1k.json" ]]; then
    DATA_FILE="./data/tinystories-1k.json"
elif [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: Training data file not found. Expected $DATA_FILE"
    echo "Please place the TinyStories JSON dataset in the data directory."
    exit 1
fi

# Set environment variables for optimal thread usage
export WALL_E_THREADS=$CPUS
export RAYON_NUM_THREADS=$CPUS

# Set environment variables for the new parameters
if [[ -n "$DATA_THREADS" ]]; then
    export WALL_E_DATA_THREADS=${DATA_THREADS#--data-threads }
fi

# Add batch timeout environment variable
if [[ -n "$BATCH_TIMEOUT" ]]; then
    export WALL_E_BATCH_TIMEOUT=${BATCH_TIMEOUT#--batch-timeout }
    echo "Setting WALL_E_BATCH_TIMEOUT=${BATCH_TIMEOUT#--batch-timeout } for multi-threaded training"
fi

# Set environment variable for multi-threaded training
if [[ -n "$MT_TRAINING" ]]; then
    export WALL_E_MT_TRAINING=1
    echo "Setting WALL_E_MT_TRAINING=1 to enable multi-threaded model training"
fi

# Configure the model save path
MODEL_SAVE_PATH="models/high_accuracy_model.walle"

# Ensure models directory exists
mkdir -p models

# Print configuration
echo "Wall-E1 Training Configuration:"
echo "-------------------------------"
echo "Model size: $SIZE"
echo "Model dimensions: $MODEL_DIM x $FF_DIM with $HEADS heads and $LAYERS layers"
echo "Training data: $DATA_FILE"
echo "Using stories: $STORIES"
echo "CPU cores: $CPUS"
echo "Epochs: $EPOCHS"
echo "Memory optimization: ${MEMORY_OPT:+Enabled}"
echo "Checkpoint strategy: ${CHECKPOINT_STRATEGY:+${CHECKPOINT_STRATEGY#--checkpoint-strategy }}"
echo "Thread optimization: ${THREAD_OPT:+${THREAD_OPT#--thread-opt }}"
echo "Batch size: ${BATCH_SIZE:+${BATCH_SIZE#--batch-size }}"
echo "Curriculum examples: $CURRICULUM_EXAMPLES"
echo "Auto-resize vocabulary: ${AUTO_RESIZE_VOCAB:+Enabled}"
echo "Watchdog timeout: ${WATCHDOG_TIMEOUT:+${WATCHDOG_TIMEOUT#--watchdog-timeout }}"
echo "Batch timeout: ${BATCH_TIMEOUT:+${BATCH_TIMEOUT#--batch-timeout }}"
echo "Data loading threads: ${DATA_THREADS:+${DATA_THREADS#--data-threads }}"
echo "Maximum target ID: ${TARGET_ID_MAX:+${TARGET_ID_MAX#--target-id-max }}"
echo "Parallel data preparation: ${PARALLEL_DATA_PREP:+Enabled}"
echo "Model save path: $MODEL_SAVE_PATH"
echo "-------------------------------"

# Run the training command
echo "Starting training..."
echo "MT_TRAINING parameter value: \"$MT_TRAINING\""

# Use the new CLI wrapper instead of directly calling cargo run
./scripts/wall-e-cli.sh \
    --model-dim $MODEL_DIM \
    --ff-dim $FF_DIM \
    --heads $HEADS \
    --layers $LAYERS \
    --epochs $EPOCHS \
    --vocab-size 5000 \
    --min-freq 2 \
    --save-path $MODEL_SAVE_PATH \
    --dataset $DATA_FILE \
    --json-format \
    --stories $STORIES \
    --learning-rate 0.0001 \
    --enable-skip \
    --strong-anti-rep \
    $MEMORY_OPT \
    $CHECKPOINT_STRATEGY \
    $THREAD_OPT \
    $BATCH_SIZE \
    --curriculum-examples $CURRICULUM_EXAMPLES \
    $AUTO_RESIZE_VOCAB \
    $WATCHDOG_TIMEOUT \
    $BATCH_TIMEOUT \
    $DATA_THREADS \
    $TARGET_ID_MAX \
    $PARALLEL_DATA_PREP \
    $MT_TRAINING

echo "Training complete. Model saved to $MODEL_SAVE_PATH"

# Generate a quick sample to show model capabilities
PROMPT="Once upon a time"
MAX_TOKENS=50

echo -e "\nGenerating sample text with prompt: \"$PROMPT\""
./scripts/test_generation.sh "$PROMPT" $MAX_TOKENS $MODEL_SAVE_PATH 

# Print debug info about the MT_TRAINING parameter
echo "MT_TRAINING value: $MT_TRAINING" 