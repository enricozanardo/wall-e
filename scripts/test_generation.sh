#!/bin/bash

# Set the prompt to generate from
PROMPT="In a world where dragons"

# Create a temporary log file
LOG_FILE="generation_log.txt"

# Run the text generation command and log output
echo "Running text generation with prompt: '$PROMPT'" | tee $LOG_FILE
echo "================================================================" | tee -a $LOG_FILE
cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.json --prompt "$PROMPT" --max-tokens 50 2>&1 | tee -a $LOG_FILE

# Check for the file output
if [ -f "generated_text.txt" ]; then
    echo "================================================================" | tee -a $LOG_FILE
    echo "Contents of generated_text.txt:" | tee -a $LOG_FILE
    cat generated_text.txt | tee -a $LOG_FILE
else
    echo "================================================================" | tee -a $LOG_FILE
    echo "generated_text.txt was not created" | tee -a $LOG_FILE
fi

# Print script completion message
echo "================================================================" | tee -a $LOG_FILE
echo "Test completed, see $LOG_FILE for full log" | tee -a $LOG_FILE 