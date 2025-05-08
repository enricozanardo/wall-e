#!/usr/bin/env python3
import json
import os
import argparse
import re

def preprocess_text(text):
    """
    Improve text formatting and word separation in stories
    """
    # Add spaces around punctuation
    text = re.sub(r'([.!?,;:()[\]{}])', r' \1 ', text)
    
    # Fix contractions
    text = re.sub(r'\s+\'([sS]|[tT]|[mM]|[rR][eE]|[vV][eE]|[lL][lL])\s+', r"'\1 ", text)
    
    # Normalize whitespace (replace multiple spaces with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add space between camelCase words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Fix quotation spacing
    text = re.sub(r'\s+"\s+', ' "', text)
    text = re.sub(r'\s+"\s+', '" ', text)
    
    return text

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Update TinyStories dataset for faster training')
    parser.add_argument('--input', default='data/tiny_stories_sample.json', help='Input dataset file path')
    parser.add_argument('--output', default='data/tiny_stories_sample_updated.json', help='Output dataset file path')
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--ff-dim', type=int, default=256, help='Feed-forward dimension')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--learning-rate', type=float, default=0.0015, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--max-seq-len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--fast', action='store_true', help='Configure for fast testing')
    parser.add_argument('--improve-text', action='store_true', help='Apply text preprocessing to improve word separation')
    args = parser.parse_args()
    
    # Adjust parameters for fast mode
    if args.fast:
        print("Fast mode enabled: Using smaller model parameters")
        args.d_model = 128
        args.ff_dim = 128
        args.num_heads = 4
        args.num_layers = 6
        args.max_seq_len = 64
        args.improve_text = True  # Always improve text in fast mode

    # Check if the dataset exists
    if not os.path.exists(args.input):
        print(f"Error: Dataset '{args.input}' not found!")
        exit(1)

    print(f"Loading dataset from '{args.input}'...")
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON data!")
        exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

    # Check if the dataset has the expected structure
    if not isinstance(data, dict) or 'stories' not in data:
        print("Error: Dataset doesn't have the expected structure!")
        exit(1)

    # Count the stories
    num_stories = len(data['stories'])
    print(f"Dataset contains {num_stories} stories")

    # Improve text formatting if requested
    if args.improve_text:
        print("Applying text preprocessing to improve word separation...")
        processed_stories = []
        for story in data['stories']:
            processed_stories.append(preprocess_text(story))
        data['stories'] = processed_stories
        print("Text preprocessing completed")

    # Calculate optimal model parameters based on dataset size and story length
    avg_story_length = sum(len(story) for story in data['stories']) / num_stories
    print(f"Average story length: {avg_story_length:.1f} characters")
    
    # Add model parameters to the dataset
    if 'model_params' not in data:
        data['model_params'] = {}
    
    # Set model parameters
    data['model_params']['d_model'] = args.d_model
    data['model_params']['ff_dim'] = args.ff_dim
    data['model_params']['num_heads'] = args.num_heads
    data['model_params']['num_layers'] = args.num_layers
    data['model_params']['dropout_rate'] = args.dropout
    data['model_params']['learning_rate'] = args.learning_rate
    data['model_params']['max_seq_len'] = args.max_seq_len
    data['model_params']['gradient_clip'] = 1.0
    data['model_params']['warmup_epochs'] = 1
    data['model_params']['plateau_epochs'] = 3
    
    # Print model parameters
    print("\nModel parameters:")
    for key, value in data['model_params'].items():
        print(f"  - {key}: {value}")
    
    # Save the updated dataset
    print(f"\nSaving updated dataset to '{args.output}'...")
    try:
        with open(args.output, 'w') as f:
            json.dump(data, f)
        print("Dataset updated successfully!")
    except Exception as e:
        print(f"Error saving dataset: {str(e)}")
        exit(1)
    
    # Print usage instructions
    print("\nUsage instructions:")
    print("1. Run the training script:")
    print("   cargo run --release --example train_story_model")
    print("\n2. For fast testing mode:")
    print("   cargo run --release --example train_story_model -- --fast")
    print("\n3. Full control:")
    print("   cargo run --release --example train_story_model -- --stories 5000 --epochs 3 --vocab-size 3000 --sample-every 1")

if __name__ == "__main__":
    main() 