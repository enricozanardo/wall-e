#!/bin/bash

# Memory profiling script for Wall-E1
# This script measures memory access patterns and cache efficiency

set -e

# Default parameters
MODEL_SIZE="small"
NUM_STORIES=100
NUM_CORES=4
DETAILED=false
OUTPUT_DIR="profiling_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --stories)
      NUM_STORIES="$2"
      shift 2
      ;;
    --cores)
      NUM_CORES="$2"
      shift 2
      ;;
    --detailed)
      DETAILED=true
      shift
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--size small|medium|large] [--stories <number>] [--cores <number>] [--detailed] [--output <dir>]"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_PREFIX="${OUTPUT_DIR}/mem_profile_${MODEL_SIZE}_${TIMESTAMP}"

echo "===== Wall-E1 Memory Profiling ====="
echo "Model size: $MODEL_SIZE"
echo "Stories: $NUM_STORIES"
echo "CPU cores: $NUM_CORES"
echo "Results will be saved to ${RESULT_PREFIX}*"

# Ensure perf is installed
if ! command -v perf &> /dev/null; then
    echo "Error: perf is not installed. Please install it with:"
    echo "  sudo apt-get install linux-tools-common linux-tools-generic"
    exit 1
fi

# Step 1: Basic statistics
echo "Running basic performance stats..."
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  ./scripts/wall-e1-model.sh train --size "$MODEL_SIZE" --stories "$NUM_STORIES" --cpus "$NUM_CORES" \
  2>&1 | tee "${RESULT_PREFIX}_stats.txt"

# Step 2: Analyze memory access patterns
echo "Analyzing memory access patterns..."
perf stat -e mem_load_retired.l3_hit,mem_load_retired.l3_miss,mem_inst_retired.all_loads,mem_inst_retired.all_stores \
  ./scripts/wall-e1-model.sh train --size "$MODEL_SIZE" --stories "$NUM_STORIES" --cpus "$NUM_CORES" \
  2>&1 | tee "${RESULT_PREFIX}_memory.txt"

# Step 3: Record call graph data for flamegraph
echo "Recording detailed profile data..."
perf record -g -F 99 \
  ./scripts/wall-e1-model.sh train --size "$MODEL_SIZE" --stories "$NUM_STORIES" --cpus "$NUM_CORES"

# Generate perf script output
perf script > "${RESULT_PREFIX}_perf.txt"

# Check if FlameGraph scripts exist, install if not
if [ ! -d "FlameGraph" ]; then
  echo "FlameGraph not found, downloading..."
  git clone https://github.com/brendangregg/FlameGraph.git
fi

# Generate flamegraph
echo "Generating flame graph..."
cat "${RESULT_PREFIX}_perf.txt" | \
  ./FlameGraph/stackcollapse-perf.pl | \
  ./FlameGraph/flamegraph.pl > "${RESULT_PREFIX}_flamegraph.svg"

echo "Basic profiling complete!"

# Step 4: Run detailed analysis if requested
if [ "$DETAILED" = true ]; then
  echo "Running detailed cache analysis with cachegrind..."
  
  # Check if valgrind is installed
  if ! command -v valgrind &> /dev/null; then
    echo "Error: valgrind is not installed. Please install it with:"
    echo "  sudo apt-get install valgrind"
    exit 1
  fi
  
  # Run with minimal dataset to avoid excessive runtime
  MINI_STORIES=$((NUM_STORIES / 10))
  if [ $MINI_STORIES -lt 1 ]; then
    MINI_STORIES=1
  fi
  
  valgrind --tool=cachegrind \
    ./scripts/wall-e1-model.sh train --size "$MODEL_SIZE" --stories "$MINI_STORIES" --cpus 1 \
    2>&1 | tee "${RESULT_PREFIX}_cachegrind.txt"
    
  # Extract cache analysis
  cg_annotate cachegrind.out.* > "${RESULT_PREFIX}_cache_report.txt"
  
  echo "Detailed cache analysis complete!"
fi

echo "===== Memory profiling complete ====="
echo "Results saved to ${RESULT_PREFIX}*"
echo
echo "Next steps:"
echo "1. Review flamegraph: ${RESULT_PREFIX}_flamegraph.svg"
echo "2. Analyze memory metrics: ${RESULT_PREFIX}_memory.txt"
echo "3. Check cache behavior: ${RESULT_PREFIX}_stats.txt"
if [ "$DETAILED" = true ]; then
  echo "4. Review detailed cache report: ${RESULT_PREFIX}_cache_report.txt"
fi 