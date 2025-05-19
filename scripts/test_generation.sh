#!/bin/bash

# Usage: ./test_generation.sh <prompt> <max_tokens> [model_path] [--cpus <num>] [--disable-watchdog]

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <prompt> <max_tokens> [model_path] [--cpus <num>] [--disable-watchdog]"
    echo "Example: $0 \"Once upon a time\" 100 models/my_model.walle"
    exit 1
fi

PROMPT="$1"
MAX_TOKENS="$2"
MODEL_PATH="${3:-models/high_accuracy_model.walle}"
CPUS=""
DISABLE_WATCHDOG=""

# Process additional arguments
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpus)
            CPUS="--cpus $2"
            shift 2
            ;;
        --disable-watchdog)
            DISABLE_WATCHDOG="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 <prompt> <max_tokens> [model_path] [--cpus <num>] [--disable-watchdog]"
            exit 1
            ;;
    esac
done

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH not found."
    echo "Available model files:"
    find models -name "*.walle" -o -name "*.bin" | sort
    exit 1
fi

echo "Using model: $MODEL_PATH"
echo "Prompt: \"$PROMPT\""
echo "Max tokens: $MAX_TOKENS"

# Set environment variables
if [ -n "$DISABLE_WATCHDOG" ]; then
    echo "Disabling watchdog for text generation"
    export WALL_E_DISABLE_WATCHDOG=true
fi

# Set up environment for better display
export RUST_LOG=info

# Use TUI version if available, otherwise use standard version
if [ -n "$CPUS" ]; then
    echo "Using $CPUS CPU cores"
    cargo run --release --bin Wall-E -- --generate-only --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --model "$MODEL_PATH" $CPUS
else
    cargo run --release --bin Wall-E -- --generate-only --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --model "$MODEL_PATH"
fi 