#!/bin/bash

# Memory optimization benchmark script for Wall-E1
# This script runs tests with different configurations to measure memory efficiency

set -e

# Default parameters
MODEL_SIZE="small"
NUM_STORIES=100
OUTPUT_DIR="benchmark_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_FILE="${OUTPUT_DIR}/memory_benchmark_${TIMESTAMP}.csv"

echo "===== Wall-E1 Memory Optimization Benchmark ====="
echo "Model size: $MODEL_SIZE"
echo "Stories: $NUM_STORIES"
echo "Results will be saved to $RESULT_FILE"

# Create CSV header
echo "Test,CPU Cores,Batch Size,Epoch Time (sec),Memory Usage (MB),Cache Misses (%)" > "$RESULT_FILE"

# Function to run a benchmark
run_benchmark() {
    local name=$1
    local cores=$2
    local batch_size=$3
    
    echo "Running benchmark: $name (cores=$cores)"
    
    # Clean before testing
    ./scripts/wall-e1-model.sh clean
    
    # Run with time and capture system stats
    TIMEFORMAT="%R"
    
    # Track memory usage with top in the background
    memory_log="${OUTPUT_DIR}/memory_${name}_${cores}.log"
    (while true; do top -b -n 1 | grep Wall-E >> "$memory_log"; sleep 0.5; done) &
    MEMORY_PID=$!
    
    # Run the actual benchmark with perf
    time_result=$( { time perf stat -e cache-references,cache-misses \
      ./scripts/wall-e1-model.sh train --size "$MODEL_SIZE" --stories "$NUM_STORIES" --cpus "$cores" \
      > /dev/null 2> "${OUTPUT_DIR}/perf_${name}_${cores}.log"; } 2>&1 )
    
    # Kill the memory monitor
    kill $MEMORY_PID 2>/dev/null || true
    
    # Extract metrics
    time_sec=$(echo "$time_result" | tr -d '\n')
    
    # Get max memory usage
    if [ -f "$memory_log" ]; then
        memory_mb=$(awk '{print $6}' "$memory_log" | sort -rn | head -1)
        # Convert from KB to MB if needed
        if [[ "$memory_mb" == *"m"* ]]; then
            memory_mb=${memory_mb%m}
        else
            memory_mb=$(echo "scale=2; $memory_mb / 1024" | bc)
        fi
    else
        memory_mb="N/A"
    fi
    
    # Extract cache misses
    perf_log="${OUTPUT_DIR}/perf_${name}_${cores}.log"
    if [ -f "$perf_log" ]; then
        cache_refs=$(grep "cache-references" "$perf_log" | awk '{print $1}' | tr -d ',')
        cache_misses=$(grep "cache-misses" "$perf_log" | awk '{print $1}' | tr -d ',')
        
        # Calculate miss ratio
        if [ "$cache_refs" != "0" ] && [ -n "$cache_refs" ]; then
            miss_ratio=$(echo "scale=2; 100 * $cache_misses / $cache_refs" | bc)
        else
            miss_ratio="N/A"
        fi
    else
        miss_ratio="N/A"
    fi
    
    # Add to CSV
    echo "$name,$cores,$batch_size,$time_sec,$memory_mb,$miss_ratio" >> "$RESULT_FILE"
    
    echo "$name: Time=${time_sec}s, Memory=${memory_mb}MB, Cache Miss=${miss_ratio}%"
    echo "----------------------------------------------------------------"
}

# Test 1: Baseline - Single core with default batch size
run_benchmark "baseline" 1 32

# Test 2: Multiple cores (half of available)
HALF_CORES=$(($(nproc) / 2))
run_benchmark "half_cores" $HALF_CORES 32

# Test 3: All cores
ALL_CORES=$(nproc)
run_benchmark "all_cores" $ALL_CORES 32

# Test 4: Memory-optimized (automatically determined by the system)
run_benchmark "memory_optimized" 0 0

# Test 5: Compare different batch sizes with optimal cores
run_benchmark "small_batch" $HALF_CORES 8
run_benchmark "medium_batch" $HALF_CORES 16
run_benchmark "large_batch" $HALF_CORES 64

echo "===== Benchmark complete ====="
echo "Results saved to $RESULT_FILE"
echo
echo "Summary:"
cat "$RESULT_FILE"
echo
echo "For detailed analysis, check the logs in the $OUTPUT_DIR directory" 