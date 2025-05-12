#!/usr/bin/env python3
import subprocess
import sys
import os

# Default prompt
prompt = "Once upon a time"
if len(sys.argv) > 1:
    prompt = sys.argv[1]

# Run the command
command = f"cargo run --release --bin train_enhanced_model -- --generate-only --model models/high_accuracy_model.json --prompt \"{prompt}\" --max-tokens 100"
print(f"Running command: {command}")

# Execute the command
try:
    result = subprocess.run(
        command, 
        shell=True, 
        check=True, 
        text=True, 
        capture_output=True
    )
    
    # Print output
    print("\n--- STDOUT ---")
    print(result.stdout)
    
    print("\n--- STDERR ---")
    print(result.stderr)
    
    # Write output to file
    with open("generation_output.txt", "w") as f:
        f.write(f"Command: {command}\n\n")
        f.write("--- STDOUT ---\n")
        f.write(result.stdout)
        f.write("\n--- STDERR ---\n")
        f.write(result.stderr)
    
    print(f"\nOutput saved to generation_output.txt")
    
except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print(f"STDOUT: {e.stdout}")
    print(f"STDERR: {e.stderr}")
except Exception as e:
    print(f"Error: {e}")

print("\nScript completed.") 