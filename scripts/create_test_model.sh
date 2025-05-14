#!/bin/bash

# Create a binary test model file for Wall-E1 with the correct format version

set -e

# Parse command line arguments
NUM_CPUS=0
output_file="models/test_binary_model.bin"

while [[ $# -gt 0 ]]; do
  case $1 in
    --output)
      output_file="$2"
      shift 2
      ;;
    --cpus)
      NUM_CPUS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--output <path>] [--cpus <number>]"
      exit 1
      ;;
  esac
done

# Set CPU Cores information
if [ "$NUM_CPUS" -gt 0 ]; then
  echo "Using $NUM_CPUS CPU cores for model creation"
  export RAYON_NUM_THREADS=$NUM_CPUS
else
  # Get available CPU cores
  AVAILABLE_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
  echo "Using all available CPU cores ($AVAILABLE_CPUS)"
  export RAYON_NUM_THREADS=$AVAILABLE_CPUS
fi

mkdir -p "$(dirname "$output_file")"

echo "Creating test binary model file at $output_file"

# Define parameters
HEADER_SIZE=64
VOCAB_OFFSET=$HEADER_SIZE

# Tokenizer parameters - updated to match train_optimized_accuracy.sh small model
VOCAB_SIZE=5000  # Match the tokenizer's vocab size from train_optimized_accuracy.sh
STRING_TABLE_SIZE=1000
TOKEN_ENTRY_SIZE=24  # 8-byte offset + 8-byte length + 8-byte token ID
NUM_TOKEN_ENTRIES=100  # Just a subset of token entries
VOCAB_SECTION_SIZE=$((16 + STRING_TABLE_SIZE + NUM_TOKEN_ENTRIES * TOKEN_ENTRY_SIZE))
WEIGHTS_OFFSET=$((VOCAB_OFFSET + VOCAB_SECTION_SIZE))

# Model architecture parameters - updated to match train_optimized_accuracy.sh small model
MODEL_DIM=128     # Embedding/model dimension
FF_DIM=512       # Feed-forward dimension
NUM_HEADS=4      # Number of attention heads
NUM_LAYERS=3     # Number of transformer layers
HEAD_DIM=$((MODEL_DIM / NUM_HEADS))

# Weight matrices:
# 1. Token embeddings [MODEL_DIM x VOCAB_SIZE]
# 2-5. Encoder layer 1:
#    - QKV projection [MODEL_DIM x (MODEL_DIM*3)]
#    - Output projection [MODEL_DIM x MODEL_DIM]
#    - FF layer 1 [MODEL_DIM x FF_DIM]
#    - FF layer 2 [FF_DIM x MODEL_DIM]
# 6-9. Encoder layer 2: same structure as layer 1
# 10-13. Encoder layer 3: same structure as layer 1
# 14. Output projection [VOCAB_SIZE x MODEL_DIM]
NUM_WEIGHT_MATRICES=$((1 + NUM_LAYERS * 4 + 1))

# Define matrix dimensions
declare -a MATRIX_ROWS
declare -a MATRIX_COLS

# 1. Token embeddings
MATRIX_ROWS[0]=$MODEL_DIM
MATRIX_COLS[0]=$VOCAB_SIZE

# Encoder layers
for ((layer=0; layer<NUM_LAYERS; layer++)); do
    # QKV projection (combined)
    MATRIX_ROWS[$((1 + layer*4))]=$MODEL_DIM
    MATRIX_COLS[$((1 + layer*4))]=$((MODEL_DIM * 3))
    
    # Output projection
    MATRIX_ROWS[$((2 + layer*4))]=$MODEL_DIM
    MATRIX_COLS[$((2 + layer*4))]=$MODEL_DIM
    
    # FF layer 1
    MATRIX_ROWS[$((3 + layer*4))]=$MODEL_DIM
    MATRIX_COLS[$((3 + layer*4))]=$FF_DIM
    
    # FF layer 2
    MATRIX_ROWS[$((4 + layer*4))]=$FF_DIM
    MATRIX_COLS[$((4 + layer*4))]=$MODEL_DIM
done

# Output projection
MATRIX_ROWS[$((1 + NUM_LAYERS*4))]=$VOCAB_SIZE
MATRIX_COLS[$((1 + NUM_LAYERS*4))]=$MODEL_DIM

# Calculate total size of all weight matrices
TOTAL_WEIGHT_SIZE=0
for ((i=0; i<NUM_WEIGHT_MATRICES; i++)); do
    MATRIX_SIZE=$((MATRIX_ROWS[i] * MATRIX_COLS[i] * 4))  # 4 bytes per float
    TOTAL_WEIGHT_SIZE=$((TOTAL_WEIGHT_SIZE + MATRIX_SIZE + 24))  # Add 24 bytes for matrix header
done

# Calculate total file size
TOTAL_FILE_SIZE=$((WEIGHTS_OFFSET + 16 + TOTAL_WEIGHT_SIZE))

# Create the file with zeros
dd if=/dev/zero of="$output_file" bs=1 count=$TOTAL_FILE_SIZE 2>/dev/null

# Write the magic bytes and version
printf '\x01\x00\x01' | dd of="$output_file" bs=1 count=3 conv=notrunc 2>/dev/null

# Write model dimensions as 64-bit little-endian values
# Model dim: 128
printf '\x80\x00\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=8 conv=notrunc 2>/dev/null

# FF dim: 512
printf '\x00\x02\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=16 conv=notrunc 2>/dev/null

# Num heads: 4
printf '\x04\x00\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=24 conv=notrunc 2>/dev/null

# Num layers: 3
printf '\x03\x00\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=32 conv=notrunc 2>/dev/null

# Vocab size: 5000
printf '\x88\x13\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=40 conv=notrunc 2>/dev/null

# Vocab offset
printf '\x40\x00\x00\x00\x00\x00\x00\x00' | dd of="$output_file" bs=1 count=8 seek=48 conv=notrunc 2>/dev/null

# Weights offset (calculated dynamically based on vocab section size)
printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
  $((WEIGHTS_OFFSET & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 8) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 16) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 24) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 32) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 40) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 48) & 0xFF)) \
  $(((WEIGHTS_OFFSET >> 56) & 0xFF)) \
  ) | dd of="$output_file" bs=1 count=8 seek=56 conv=notrunc 2>/dev/null

# Create vocabulary section (starts at byte 64)
# Actual vocab size
printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
  $((VOCAB_SIZE & 0xFF)) \
  $(((VOCAB_SIZE >> 8) & 0xFF)) \
  $(((VOCAB_SIZE >> 16) & 0xFF)) \
  $(((VOCAB_SIZE >> 24) & 0xFF)) \
  $(((VOCAB_SIZE >> 32) & 0xFF)) \
  $(((VOCAB_SIZE >> 40) & 0xFF)) \
  $(((VOCAB_SIZE >> 48) & 0xFF)) \
  $(((VOCAB_SIZE >> 56) & 0xFF)) \
  ) | dd of="$output_file" bs=1 count=8 seek=$VOCAB_OFFSET conv=notrunc 2>/dev/null

# String table size
printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
  $((STRING_TABLE_SIZE & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 8) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 16) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 24) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 32) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 40) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 48) & 0xFF)) \
  $(((STRING_TABLE_SIZE >> 56) & 0xFF)) \
  ) | dd of="$output_file" bs=1 count=8 seek=$((VOCAB_OFFSET+8)) conv=notrunc 2>/dev/null

# Add some dummy string data in the string table
echo -n "token0token1token2token3token4token5" | dd of="$output_file" bs=1 count=$STRING_TABLE_SIZE seek=$((VOCAB_OFFSET+16)) conv=notrunc 2>/dev/null

# Add number of weight matrices
printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
  $((NUM_WEIGHT_MATRICES & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 8) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 16) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 24) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 32) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 40) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 48) & 0xFF)) \
  $(((NUM_WEIGHT_MATRICES >> 56) & 0xFF)) \
  ) | dd of="$output_file" bs=1 count=8 seek=$WEIGHTS_OFFSET conv=notrunc 2>/dev/null

# Total weight size
printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
  $((TOTAL_WEIGHT_SIZE & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 8) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 16) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 24) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 32) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 40) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 48) & 0xFF)) \
  $(((TOTAL_WEIGHT_SIZE >> 56) & 0xFF)) \
  ) | dd of="$output_file" bs=1 count=8 seek=$((WEIGHTS_OFFSET+8)) conv=notrunc 2>/dev/null

# Current offset for weight data
CURRENT_OFFSET=$((WEIGHTS_OFFSET + 16))

# Add matrix headers and data for each weight matrix
for ((i=0; i<NUM_WEIGHT_MATRICES; i++)); do
    MATRIX_ROWS_VAL=${MATRIX_ROWS[i]}
    MATRIX_COLS_VAL=${MATRIX_COLS[i]}
    MATRIX_SIZE=$((MATRIX_ROWS_VAL * MATRIX_COLS_VAL * 4))  # 4 bytes per float
    
    # Write matrix dimensions
    printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
      $((MATRIX_ROWS_VAL & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 8) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 16) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 24) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 32) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 40) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 48) & 0xFF)) \
      $(((MATRIX_ROWS_VAL >> 56) & 0xFF)) \
      ) | dd of="$output_file" bs=1 count=8 seek=$CURRENT_OFFSET conv=notrunc 2>/dev/null
    CURRENT_OFFSET=$((CURRENT_OFFSET + 8))
    
    printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
      $((MATRIX_COLS_VAL & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 8) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 16) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 24) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 32) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 40) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 48) & 0xFF)) \
      $(((MATRIX_COLS_VAL >> 56) & 0xFF)) \
      ) | dd of="$output_file" bs=1 count=8 seek=$CURRENT_OFFSET conv=notrunc 2>/dev/null
    CURRENT_OFFSET=$((CURRENT_OFFSET + 8))
    
    # Write matrix size
    printf $(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x\\x%02x' \
      $((MATRIX_SIZE & 0xFF)) \
      $(((MATRIX_SIZE >> 8) & 0xFF)) \
      $(((MATRIX_SIZE >> 16) & 0xFF)) \
      $(((MATRIX_SIZE >> 24) & 0xFF)) \
      $(((MATRIX_SIZE >> 32) & 0xFF)) \
      $(((MATRIX_SIZE >> 40) & 0xFF)) \
      $(((MATRIX_SIZE >> 48) & 0xFF)) \
      $(((MATRIX_SIZE >> 56) & 0xFF)) \
      ) | dd of="$output_file" bs=1 count=8 seek=$CURRENT_OFFSET conv=notrunc 2>/dev/null
    CURRENT_OFFSET=$((CURRENT_OFFSET + 8))
    
    # Create controlled random float values for the matrix
    # Use seeded randomness for testing so results are reproducible
    for ((r=0; r<MATRIX_ROWS_VAL; r++)); do
        for ((c=0; c<MATRIX_COLS_VAL; c++)); {
            # Generate a value between -0.5 and 0.5 based on position
            SEED=$((r * 1000 + c + i * 10000))
            VALUE=$(echo "scale=6; (($SEED % 1000) / 1000.0) - 0.5" | bc)
            
            # Convert VALUE to IEEE 754 binary representation (simplified)
            # This is a very basic approximation - in a real scenario use proper conversion
            printf "%.6f" $VALUE | xxd -p | dd of="$output_file" bs=1 count=4 seek=$CURRENT_OFFSET conv=notrunc 2>/dev/null
            CURRENT_OFFSET=$((CURRENT_OFFSET + 4))
        }
    done
    
    echo "Created matrix $((i+1))/$NUM_WEIGHT_MATRICES: ${MATRIX_ROWS_VAL}x${MATRIX_COLS_VAL}"
done

echo "Test binary model file created successfully"
echo "File size: $(wc -c < "$output_file") bytes"
echo "Model configuration: dim=$MODEL_DIM, ff_dim=$FF_DIM, heads=$NUM_HEADS, layers=$NUM_LAYERS"
echo "Number of weight matrices: $NUM_WEIGHT_MATRICES"
echo "Using RAYON_NUM_THREADS=$RAYON_NUM_THREADS CPU cores"

# Make the script executable
chmod +x "$0" 