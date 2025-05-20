#!/bin/bash

# Enable verbose output
set -ex

# Set the batch timeout for multi-threaded training
export WALL_E_BATCH_TIMEOUT=30

echo "==============================================="
echo "Testing Multi-Threaded Training with small model"
echo "==============================================="

# Run with multi-threaded training enabled
./scripts/wall-e1-model.sh train --size small --stories 500 --epochs 1 --mt-training --batch-timeout 60

# Check CPU usage when running the actual model
echo "==============================================="
echo "Monitoring CPU usage with MT training..."
echo "==============================================="

# Run the CPU monitor in the background
(
  while true; do
    top -b -n 1 | grep "Cpu(s)" | awk '{print $2 " " $4}'
    sleep 1
  done
) > cpu_usage.log &
CPU_MONITOR_PID=$!

# Run the test_multithreading binary
cargo run --release --bin test_multithreading

# Kill the CPU monitor
kill $CPU_MONITOR_PID

# Show CPU usage
echo "CPU Usage during test:"
cat cpu_usage.log

echo "===============================================" 
echo "Test completed successfully"
echo "===============================================" 