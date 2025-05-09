#!/bin/bash
# Optimized training script based on initial results analysis
# Designed for better accuracy in less time with more efficient parameters

echo "===== WALL-E1 OPTIMIZED TRAINING SCRIPT (FAST & EFFICIENT) ====="
echo "This script uses parameters optimized for better generalization and speed"

# Create models directory if it doesn't exist
mkdir -p models

# Set environment variables for better threading
export RAYON_NUM_THREADS=16

# Run the training with optimized parameters
echo "Starting training with efficient parameters for improved accuracy..."
cargo run --release --example train_with_improved_tokenization -- \
  --stories 8000 \
  --vocab-size 5000 \
  --epochs 8 \
  --model-dim 192 \
  --ff-dim 384 \
  --heads 6 \
  --layers 4 \
  --rep-penalty 2.0 \
  --pres-penalty 0.6 \
  --freq-penalty 0.6 \
  --curriculum-step 1 \
  --batch-size 32

echo "===== TRAINING COMPLETE ====="
echo "The final model is saved in models/improved_model.bin"
echo "You can test the model quality with: cargo run --release --example test_model" 