#!/bin/bash
# Run a full training with optimal parameters for high accuracy

echo "===== WALL-E1 OPTIMIZED TRAINING SCRIPT ====="
echo "This will train the model with parameters optimized for >80% accuracy"

# Create models directory if it doesn't exist
mkdir -p models

# Set environment variables for better threading
export RAYON_NUM_THREADS=4

# Run the training with optimized parameters
echo "Starting training with optimal parameters for high accuracy..."
cargo run --release --example train_with_improved_tokenization -- \
  --stories 15000 \
  --vocab-size 8000 \
  --epochs 12 \
  --model-dim 384 \
  --ff-dim 768 \
  --heads 12 \
  --layers 8

echo "===== TRAINING COMPLETE ====="
echo "The final model is saved in models/improved_model.bin"
echo "You can test the model quality with: cargo run --release --example test_model" 