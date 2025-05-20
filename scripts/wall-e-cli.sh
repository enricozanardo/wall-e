#!/bin/bash

# Wall-E CLI Wrapper
# This script serves as a bridge between wall-e1-model.sh and the Wall-E binary,
# translating parameters to the correct format

set -e

# Display usage information
show_usage() {
  echo "Wall-E CLI Wrapper"
  echo "Usage: $0 [options]"
  echo "This script translates parameters between wall-e1-model.sh and the Wall-E binary."
  echo "It should not be called directly - use wall-e1-model.sh instead."
}

# DATASET FISSO - elimina la necessità di passare il parametro
FIXED_DATA_PATH="./data/tiny_stories_sample.json"
if [[ -f "./data/tinystories-10k.json" ]]; then
    FIXED_DATA_PATH="./data/tinystories-10k.json"
elif [[ -f "./data/tinystories-1k.json" ]]; then
    FIXED_DATA_PATH="./data/tinystories-1k.json"
fi

echo "USANDO DATASET FISSO: $FIXED_DATA_PATH"

# Initialize parameters with correct names for the binary
MODEL_DIM=""
FF_DIM=""
HEADS=""
LAYERS=""
DROPOUT=""
EPOCHS=""
VOCAB_SIZE=""
MIN_FREQ=""
LEARNING_RATE=""
SAVE_PATH=""
STORIES=""
ENABLE_SKIP=""
DISABLE_CURRICULUM=""
STRONG_ANTI_REP=""
JSON_FORMAT="--json-format"  # Attivato di default
AUTO_RESIZE_VOCAB=""
MEMORY_OPT=""
PARALLEL=""
MT_TRAINING=""
WATCHDOG_TIMEOUT=""
BATCH_TIMEOUT=""
DATA_THREADS=""
TARGET_ID_MAX=""
CHECKPOINT_STRATEGY=""
THREAD_OPT=""
BATCH_SIZE=""
CURRICULUM_EXAMPLES=""

# Process and translate arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model-dim)
            MODEL_DIM="--model-dim $2"
            shift 2
            ;;
        --ff-dim)
            FF_DIM="--ff-dim $2"
            shift 2
            ;;
        --heads)
            HEADS="--heads $2"
            shift 2
            ;;
        --layers)
            LAYERS="--layers $2"
            shift 2
            ;;
        --dropout)
            DROPOUT="--dropout $2"
            shift 2
            ;;
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="--vocab-size $2"
            shift 2
            ;;
        --min-freq)
            MIN_FREQ="--min-freq $2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="--learning-rate $2"
            shift 2
            ;;
        --save-path)
            SAVE_PATH="--save-path $2"
            shift 2
            ;;
        --stories)
            STORIES="--stories $2"
            shift 2
            ;;
        --enable-skip)
            ENABLE_SKIP="--enable-skip"
            shift
            ;;
        --disable-curriculum)
            DISABLE_CURRICULUM="--disable-curriculum"
            shift
            ;;
        --strong-anti-rep)
            STRONG_ANTI_REP="--strong-anti-rep"
            shift
            ;;
        --json-format)
            # Già attivato per default
            shift
            ;;
        --auto-resize-vocab)
            AUTO_RESIZE_VOCAB="--auto-resize-vocab yes"
            shift
            ;;
        --memory-opt)
            MEMORY_OPT="--memory-opt"
            shift
            ;;
        --parallel)
            PARALLEL="--parallel"
            shift
            ;;
        --mt-training)
            MT_TRAINING="--mt-training"
            # Also set the environment variable
            export WALL_E_MT_TRAINING=1
            shift
            ;;
        --watchdog-timeout)
            WATCHDOG_TIMEOUT="--watchdog-timeout $2"
            shift 2
            ;;
        --batch-timeout)
            BATCH_TIMEOUT="--batch-timeout $2"
            # Also set the environment variable
            export WALL_E_BATCH_TIMEOUT=$2
            shift 2
            ;;
        --data-threads)
            DATA_THREADS="--data-threads $2"
            # Also set the environment variable
            export WALL_E_DATA_THREADS=$2
            shift 2
            ;;
        --target-id-max)
            TARGET_ID_MAX="--target-id-max $2"
            shift 2
            ;;
        --checkpoint-strategy)
            CHECKPOINT_STRATEGY="--checkpoint $2"
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
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Set RUST_BACKTRACE for better error reporting
export RUST_BACKTRACE=1

# Messaggio appropriato in base al tipo di training
if [[ -n "$MT_TRAINING" ]]; then
    echo "Using the Wall-E binary with multi-threaded training enabled."
else
    echo "Using the Wall-E binary for standard training."
fi

# Add the missing --dataset parameter to command line
echo "Passing --dataset $FIXED_DATA_PATH to Wall-E"

# Execute the Wall-E binary with the translated parameters
cargo run --release --bin Wall-E -- \
    $MODEL_DIM \
    $FF_DIM \
    $HEADS \
    $LAYERS \
    $DROPOUT \
    $EPOCHS \
    $VOCAB_SIZE \
    $MIN_FREQ \
    $LEARNING_RATE \
    $SAVE_PATH \
    $STORIES \
    $ENABLE_SKIP \
    $DISABLE_CURRICULUM \
    $STRONG_ANTI_REP \
    $JSON_FORMAT \
    $AUTO_RESIZE_VOCAB \
    $MEMORY_OPT \
    $PARALLEL \
    $MT_TRAINING \
    $WATCHDOG_TIMEOUT \
    $BATCH_TIMEOUT \
    $DATA_THREADS \
    $TARGET_ID_MAX \
    $CHECKPOINT_STRATEGY \
    $THREAD_OPT \
    $BATCH_SIZE \
    $CURRICULUM_EXAMPLES \
    --dataset $FIXED_DATA_PATH