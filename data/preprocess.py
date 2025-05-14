import json
import re
import sys
from pathlib import Path

def clean_text(text):
    # Normalize spacing around punctuation
    text = re.sub(r'([.,!?:;])', r' \1 ', text)
    # Fix contractions
    text = re.sub(r"(\w)'(\w)", r"\1' \2", text)  
    # Normalize quotes
    text = re.sub(r'[""]', ' " ', text)
    text = re.sub(r'['']', " ' ", text)
    # Remove duplicate spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stories = data.get('stories', [])
    cleaned_stories = []
    
    for story in stories:
        if isinstance(story, str):
            text = story
        elif isinstance(story, dict) and 'text' in story:
            text = story['text']
        else:
            continue
        
        cleaned_text = clean_text(text)
        cleaned_stories.append(cleaned_text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'stories': cleaned_stories}, f)
    
    print(f'Processed {len(cleaned_stories)} stories')
    return len(cleaned_stories)

def process_text_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    cleaned_text = clean_text(text)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    print(f'Processed {len(cleaned_text)} characters of text')
    return len(cleaned_text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocess.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if input_path.endswith('.json'):
        process_json_file(input_path, output_path)
    else:
        process_text_file(input_path, output_path)
