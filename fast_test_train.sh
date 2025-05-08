#!/bin/bash

# Script to quickly test the training process with reduced data

echo "=== Wall-E1 Fast Model Testing with Word Separation Improvements ==="
echo "This script will run a quick training test with improved word separation"

# First update the dataset with fast mode
echo -e "\n=== Updating dataset with fast mode parameters and text improvements ==="
python3 update_tiny_stories.py --fast --improve-text

# Create the models directory if it doesn't exist
mkdir -p models

# Run the training script with fast mode
echo -e "\n=== Running training in fast mode ==="
cargo run --release --example train_story_model -- --fast --stories 1000 --epochs 15 --sample-every 1 --temperature 0.7

echo -e "\n=== Test completed ==="
echo "Check the generated text samples above to evaluate model quality." 