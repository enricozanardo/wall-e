#!/usr/bin/env python3
"""
Helper script for text generation with the Wall-E1 model.
Provides a nice command-line interface with more formatting options.
"""

import argparse
import subprocess
import os
import sys
import tempfile

def colorize(text, color_code):
    """Apply color to terminal output."""
    return f"\033[{color_code}m{text}\033[0m"

def generate_text(model_path, prompt, max_tokens=50, verbose=False):
    """Generate text using the Wall-E1 model."""
    print(f"Generating text with prompt: {colorize(prompt, '1;32')}")
    print(f"Using model: {colorize(model_path, '1;34')}")
    print(f"Max tokens: {colorize(str(max_tokens), '1;34')}")
    
    # Ensure the model file exists
    if not os.path.exists(model_path):
        print(colorize(f"Error: Model file not found: {model_path}", "1;31"))
        print("Please run the training script first:")
        print(colorize("./scripts/train_optimized_accuracy.sh --size small", "1;33"))
        return False
    
    # Run the command with error handling
    try:
        cmd = [
            "cargo", "run", "--release", "--bin", "train_enhanced_model", "--",
            "--generate-only",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens)
        ]
        
        if verbose:
            print(colorize("Running command: " + " ".join(cmd), "1;30"))
        
        # Run the command and capture output
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Print the output
        output_lines = result.stdout.split("\n")
        for line in output_lines:
            if "Generated preview:" in line:
                preview = line.split("Generated preview:")[1].strip()
                print(f"\nGenerated preview: {colorize(preview, '1;36')}")
            
            # Look for the generated text section
            if "======= GENERATED TEXT =======" in line:
                start_index = output_lines.index(line)
                if start_index + 2 < len(output_lines):
                    generated_text = output_lines[start_index + 1]
                    print(f"\n{colorize('GENERATED TEXT:', '1;35')}")
                    print(f"{colorize(generated_text, '1;37')}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(colorize(f"Error running generation command: {e}", "1;31"))
        if e.stderr:
            print(colorize("Error details:", "1;31"))
            print(e.stderr)
        return False
    except Exception as e:
        print(colorize(f"Unexpected error: {e}", "1;31"))
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate text using the Wall-E1 model")
    parser.add_argument("prompt", help="Prompt to start the text generation")
    parser.add_argument("--model", default="models/high_accuracy_model.walle", 
                       help="Path to the model file (default: models/high_accuracy_model.walle)")
    parser.add_argument("--max-tokens", type=int, default=50, 
                       help="Maximum number of tokens to generate (default: 50)")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    print(colorize("=== Wall-E1 Text Generation ===", "1;33"))
    success = generate_text(args.model, args.prompt, args.max_tokens, args.verbose)
    
    if success:
        print(colorize("\nText generation complete!", "1;32"))
    else:
        print(colorize("\nText generation failed.", "1;31"))
        sys.exit(1)

if __name__ == "__main__":
    main() 